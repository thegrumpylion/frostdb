package frostdb

import (
	"bytes"
	"context"
	"errors"
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/go-kit/log/level"

	"github.com/polarsignals/frostdb/dynparquet"
	walpb "github.com/polarsignals/frostdb/gen/proto/go/frostdb/wal/v1alpha1"
)

// ErrTransactionFinished is returned when a Transaction is used after
// Commit or Abort.
var ErrTransactionFinished = errors.New("transaction already committed or aborted")

// TableTx is a Transaction's handle to a single table.
type TableTx interface {
	// InsertRecord inserts an arrow record into the table as part of the
	// transaction. The rows are invisible to readers and not durable until
	// the transaction commits. An error poisons the transaction: every
	// subsequent operation fails and Commit aborts.
	InsertRecord(context.Context, arrow.Record) error
}

// Transaction is a multi-table write transaction. All inserts made through
// it become visible to readers atomically when Commit returns, or never if
// it is aborted (explicitly, or implicitly by a crash before commit — an
// uncommitted transaction writes nothing to the WAL).
//
// A Transaction is not safe for concurrent use.
//
// Every Transaction MUST end in exactly one Commit or Abort call. The
// database's high watermark — the read-visibility horizon for ALL tables —
// cannot advance past an open transaction, so a leaked transaction blocks
// read visibility of every later write database-wide and eventually blocks
// Close. Keep transactions short-lived; insert pre-built records and
// commit. The leak-proof usage pattern is to defer an Abort immediately
// after Begin (Abort after Commit is a no-op), so early error returns —
// including a failed GetTable — can never leak the transaction:
//
//	tx := db.Begin()
//	defer tx.Abort()
//	...
//	return tx.Commit()
type Transaction interface {
	// GetTable returns the transaction's handle to the named table. The
	// table must already exist (see DB.Table); a lookup failure does not
	// poison the transaction.
	GetTable(name string) (TableTx, error)
	// Commit makes the transaction's inserts visible and durable as one
	// atomic unit. On error the transaction was aborted instead: none of
	// its inserts are visible or durable.
	Commit() error
	// Abort discards the transaction's inserts. Safe to call after a
	// failed Commit; a no-op once the transaction is finished.
	Abort()
}

// Begin starts a multi-table write transaction. See Transaction for the
// visibility, durability, and lifetime contract.
func (db *DB) Begin() Transaction {
	tx, _, complete := db.begin()
	return &transaction{
		db:       db,
		tx:       tx,
		complete: complete,
		blocks:   map[*TableBlock]struct{}{},
	}
}

type transaction struct {
	db *DB

	// tx is the single transaction id shared by every insert in this
	// transaction; complete releases its watermark slot.
	tx       uint64
	complete func()
	finished bool

	// err poisons the transaction after a failed insert: some tables may
	// already hold parts for tx, so the only sound outcomes are abort.
	err error

	// writes accumulates one serialized WAL sub-entry per insert. They are
	// logged as a single atomic WAL entry at commit; an aborted
	// transaction logs a TransactionAborted entry instead (the WAL queue
	// needs exactly one entry per allocated tx id to make progress).
	writes []*walpb.Entry_Write
	// blocks is the set of table blocks holding this transaction's parts.
	// Abort removes the parts from each block's index. Tracking the block
	// per insert (rather than the table's active block at abort time)
	// matters: a rotation between insert and abort moves the parts into a
	// pending block, and the watermark gate in writeBlock keeps that block
	// unpersisted until this transaction finishes.
	blocks map[*TableBlock]struct{}
}

type tableTx struct {
	tx    *transaction
	table *Table
}

// poison records the first insert failure. The transaction may already
// hold parts for tx in some tables, so it can no longer commit: Commit
// aborts and returns the recorded error.
func (t *transaction) poison(err error) error {
	if t.err == nil {
		t.err = err
	}
	return err
}

func (t *transaction) GetTable(name string) (TableTx, error) {
	if t.finished {
		return nil, ErrTransactionFinished
	}
	if t.err != nil {
		return nil, t.err
	}
	table, err := t.db.GetTable(name)
	if err != nil {
		return nil, err
	}
	return &tableTx{tx: t, table: table}, nil
}

func (t *tableTx) InsertRecord(ctx context.Context, record arrow.Record) error {
	if t.tx.finished {
		return ErrTransactionFinished
	}
	if t.tx.err != nil {
		return t.tx.err
	}

	block, finish, err := t.table.appender(ctx)
	if err != nil {
		return t.tx.poison(fmt.Errorf("get appender: %w", err))
	}
	defer finish()

	preHashed := dynparquet.PrehashColumns(t.table.schema, record)
	defer preHashed.Release()

	// Serialize for the commit-time WAL entry before touching the index,
	// so a serialization failure leaves no state behind for this insert.
	var walWrite *walpb.Entry_Write
	if !t.table.config.Load().DisableWal {
		var buf bytes.Buffer
		if err := func() error {
			w := ipc.NewWriter(&buf, ipc.WithSchema(preHashed.Schema()))
			defer w.Close()
			return w.Write(preHashed)
		}(); err != nil {
			return t.tx.poison(fmt.Errorf("serialize record for WAL: %w", err))
		}
		walWrite = &walpb.Entry_Write{
			TableName: t.table.name,
			Data:      buf.Bytes(),
			Arrow:     true,
		}
	}

	if err := block.InsertRecord(ctx, t.tx.tx, preHashed); err != nil {
		return t.tx.poison(fmt.Errorf("insert into block: %w", err))
	}

	if walWrite != nil {
		t.tx.writes = append(t.tx.writes, walWrite)
	}
	t.tx.blocks[block] = struct{}{}
	return nil
}

func (t *transaction) Commit() error {
	if t.finished {
		return ErrTransactionFinished
	}
	if t.err != nil {
		err := t.err
		t.abort()
		return err
	}

	if err := t.db.wal.Log(t.tx, &walpb.Record{
		Entry: &walpb.Entry{
			EntryType: &walpb.Entry_TransactionWrite_{
				TransactionWrite: &walpb.Entry_TransactionWrite{
					Writes: t.writes,
				},
			},
		},
	}); err != nil {
		t.err = fmt.Errorf("log transaction: %w", err)
		t.abort()
		return t.err
	}

	t.finish()
	return nil
}

func (t *transaction) Abort() {
	if t.finished {
		return
	}
	t.abort()
}

// abort removes the transaction's parts from every block it wrote to,
// fills the transaction's WAL slot with an aborted entry, and completes
// the transaction. Removal happens BEFORE complete: the watermark may only
// pass this transaction once its parts are gone, which is what lets block
// persistence (gated on the watermark in writeBlock) trust that every part
// it sees belongs to a finished transaction.
func (t *transaction) abort() {
	for block := range t.blocks {
		block.index.Remove(t.tx)
	}
	if err := t.db.wal.Log(t.tx, &walpb.Record{
		Entry: &walpb.Entry{
			EntryType: &walpb.Entry_TransactionAborted_{
				TransactionAborted: &walpb.Entry_TransactionAborted{},
			},
		},
	}); err != nil {
		// Failing to fill the tx slot stalls WAL progress (the queue waits
		// for this id). Log can only fail before enqueueing, so surface it
		// loudly; nothing else can be done here.
		level.Error(t.db.logger).Log("msg", "failed to log transaction abort; WAL may stall", "tx", t.tx, "err", err)
	}
	t.finish()
}

func (t *transaction) finish() {
	t.finished = true
	t.complete()
	t.writes = nil
	t.blocks = nil
}
