package frostdb

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/require"
	"github.com/thanos-io/objstore"

	"github.com/polarsignals/frostdb/dynparquet"
	"github.com/polarsignals/frostdb/query"
)

// countRows scans the named table through the query engine and returns the
// number of visible rows.
func countRows(t *testing.T, db *DB, table string) int64 {
	t.Helper()
	pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
	defer pool.AssertSize(t, 0)
	rows := int64(0)
	engine := query.NewEngine(pool, db.TableProvider())
	err := engine.ScanTable(table).
		Execute(context.Background(), func(_ context.Context, r arrow.Record) error {
			rows += r.NumRows()
			return nil
		})
	require.NoError(t, err)
	return rows
}

func testSamplesRecord(t *testing.T, n int) arrow.Record {
	t.Helper()
	samples := dynparquet.GenerateTestSamples(n)
	r, err := samples.ToRecord()
	require.NoError(t, err)
	return r
}

// Test_Transaction_Visibility pins the in-memory transaction contract:
// inserts are invisible until Commit, visible atomically across tables
// after it, and never visible after Abort.
func Test_Transaction_Visibility(t *testing.T) {
	tests := map[string]func(t *testing.T, db *DB, r arrow.Record){
		"read isolation and commit": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))

			require.Equal(t, int64(3), countRows(t, db, "test"))
			require.NoError(t, tx.Commit())
			require.Equal(t, int64(6), countRows(t, db, "test"))
		},
		"abort": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))

			require.Equal(t, int64(3), countRows(t, db, "test"))
			tx.Abort()
			require.Equal(t, int64(3), countRows(t, db, "test"))
		},
		"multi table commit": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))
			tbl2, err := tx.GetTable("test2")
			require.NoError(t, err)
			require.NoError(t, tbl2.InsertRecord(ctx, r))

			require.Equal(t, int64(3), countRows(t, db, "test"))
			require.Equal(t, int64(3), countRows(t, db, "test2"))
			require.NoError(t, tx.Commit())
			require.Equal(t, int64(6), countRows(t, db, "test"))
			require.Equal(t, int64(6), countRows(t, db, "test2"))
		},
		"multi table abort": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))
			tbl2, err := tx.GetTable("test2")
			require.NoError(t, err)
			require.NoError(t, tbl2.InsertRecord(ctx, r))

			tx.Abort()
			require.Equal(t, int64(3), countRows(t, db, "test"))
			require.Equal(t, int64(3), countRows(t, db, "test2"))
		},
		"multiple inserts same table aborted": func(t *testing.T, db *DB, r arrow.Record) {
			// Adjacent same-tx parts are the LSM.Remove stress case.
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))
			require.NoError(t, tbl.InsertRecord(ctx, r))
			require.NoError(t, tbl.InsertRecord(ctx, r))

			tx.Abort()
			require.Equal(t, int64(3), countRows(t, db, "test"))
		},
		"later single-table write stalls until commit": func(t *testing.T, db *DB, r arrow.Record) {
			// The documented watermark stall: a single-table write that
			// completes AFTER an open transaction began is invisible until
			// the transaction finishes, then both become visible.
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))

			table, err := db.GetTable("test2")
			require.NoError(t, err)
			_, err = table.InsertRecord(ctx, r)
			require.NoError(t, err)

			require.Equal(t, int64(3), countRows(t, db, "test2"))
			require.NoError(t, tx.Commit())
			require.Equal(t, int64(6), countRows(t, db, "test"))
			require.Equal(t, int64(6), countRows(t, db, "test2"))
		},
		"use after finish": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))
			require.NoError(t, tx.Commit())

			require.ErrorIs(t, tbl.InsertRecord(ctx, r), ErrTransactionFinished)
			_, err = tx.GetTable("test")
			require.ErrorIs(t, err, ErrTransactionFinished)
			require.ErrorIs(t, tx.Commit(), ErrTransactionFinished)
			tx.Abort() // no-op, must not corrupt the watermark
			require.Equal(t, int64(6), countRows(t, db, "test"))

			// The watermark must still advance: a subsequent write is
			// visible.
			table, err := db.GetTable("test")
			require.NoError(t, err)
			_, err = table.InsertRecord(ctx, r)
			require.NoError(t, err)
			require.Equal(t, int64(9), countRows(t, db, "test"))
		},
		"poisoned transaction aborts on commit": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))

			sentinel := errors.New("sentinel insert failure")
			tx.(*transaction).poison(sentinel)

			require.ErrorIs(t, tbl.InsertRecord(ctx, r), sentinel)
			require.ErrorIs(t, tx.Commit(), sentinel)
			// The pre-poison insert was rolled back with the abort.
			require.Equal(t, int64(3), countRows(t, db, "test"))
		},
		"empty transaction commit": func(t *testing.T, db *DB, r arrow.Record) {
			ctx := context.Background()
			require.NoError(t, db.Begin().Commit())
			// The watermark advanced past the empty transaction: a
			// subsequent write is visible.
			table, err := db.GetTable("test")
			require.NoError(t, err)
			_, err = table.InsertRecord(ctx, r)
			require.NoError(t, err)
			require.Equal(t, int64(6), countRows(t, db, "test"))
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			c, err := New(
				WithLogger(newTestLogger(t)),
				WithActiveMemorySize(100*KiB),
			)
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, c.Close()) })
			db, err := c.DB(context.Background(), "test")
			require.NoError(t, err)
			_, err = db.Table("test", NewTableConfig(dynparquet.SampleDefinition()))
			require.NoError(t, err)
			_, err = db.Table("test2", NewTableConfig(dynparquet.SampleDefinition()))
			require.NoError(t, err)

			r := testSamplesRecord(t, 3)
			defer r.Release()

			// Seed both tables with 3 rows outside any transaction.
			ctx := context.Background()
			for _, name := range []string{"test", "test2"} {
				table, err := db.GetTable(name)
				require.NoError(t, err)
				_, err = table.InsertRecord(ctx, r)
				require.NoError(t, err)
			}

			test(t, db, r)
		})
	}
}

// Test_Transaction_WALReplay pins the durability contract: a committed
// transaction replays atomically across tables; an aborted transaction
// leaves nothing to replay AND does not stall WAL progress for later
// writes (its tx slot is filled by the aborted entry).
func Test_Transaction_WALReplay(t *testing.T) {
	for _, committed := range []bool{true, false} {
		name := "aborted"
		if committed {
			name = "committed"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			logger := newTestLogger(t)

			open := func() (*ColumnStore, *DB) {
				c, err := New(
					WithLogger(logger),
					WithWAL(),
					WithStoragePath(dir),
				)
				require.NoError(t, err)
				db, err := c.DB(ctx, "test")
				require.NoError(t, err)
				for _, name := range []string{"test", "test2"} {
					_, err = db.Table(name, NewTableConfig(dynparquet.SampleDefinition()))
					require.NoError(t, err)
				}
				return c, db
			}

			c, db := open()
			r := testSamplesRecord(t, 3)
			defer r.Release()

			tx := db.Begin()
			for _, name := range []string{"test", "test2"} {
				tbl, err := tx.GetTable(name)
				require.NoError(t, err)
				require.NoError(t, tbl.InsertRecord(ctx, r))
			}
			if committed {
				require.NoError(t, tx.Commit())
			} else {
				tx.Abort()
			}

			// A later single-table write must be durable in both cases:
			// if the aborted transaction's WAL slot were not filled, the
			// WAL queue would stall and this write would never persist.
			table, err := db.GetTable("test")
			require.NoError(t, err)
			_, err = table.InsertRecord(ctx, r)
			require.NoError(t, err)

			require.NoError(t, c.Close())

			c, db = open()
			t.Cleanup(func() { require.NoError(t, c.Close()) })

			if committed {
				require.Equal(t, int64(6), countRows(t, db, "test"))
				require.Equal(t, int64(3), countRows(t, db, "test2"))
			} else {
				require.Equal(t, int64(3), countRows(t, db, "test"))
				require.Equal(t, int64(0), countRows(t, db, "test2"))
			}
		})
	}
}

// Test_Transaction_RotationDuringOpenTx pins the block-persistence gate:
// a block rotation while a transaction is open must not persist the
// transaction's parts. writeBlock waits for the watermark to cover the
// block's max tx, so an abort removes its parts before they can reach
// parquet, and a commit persists them.
func Test_Transaction_RotationDuringOpenTx(t *testing.T) {
	for _, committed := range []bool{true, false} {
		name := "aborted"
		if committed {
			name = "committed"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			dir := t.TempDir()
			logger := newTestLogger(t)
			bucket := objstore.NewInMemBucket()

			open := func() (*ColumnStore, *DB) {
				c, err := New(
					WithLogger(logger),
					WithWAL(),
					WithStoragePath(dir),
					WithReadWriteStorage(NewDefaultObjstoreBucket(bucket)),
				)
				require.NoError(t, err)
				db, err := c.DB(ctx, "test")
				require.NoError(t, err)
				_, err = db.Table("test", NewTableConfig(dynparquet.SampleDefinition()))
				require.NoError(t, err)
				return c, db
			}

			c, db := open()
			r := testSamplesRecord(t, 3)
			defer r.Release()

			// Seed a committed row set so the rotated block is non-empty
			// even after an abort.
			table, err := db.GetTable("test")
			require.NoError(t, err)
			_, err = table.InsertRecord(ctx, r)
			require.NoError(t, err)

			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			require.NoError(t, err)
			require.NoError(t, tbl.InsertRecord(ctx, r))

			// Rotate the active block while the transaction is open. The
			// persist goroutine must block on the watermark until the
			// transaction finishes.
			var wg sync.WaitGroup
			wg.Add(1)
			require.NoError(t, table.RotateBlock(ctx, table.ActiveBlock(), WithRotateBlockWaitGroup(&wg)))

			// Nothing may reach the bucket while the transaction is open:
			// the watermark gate in writeBlock is the only thing holding
			// the persist goroutine back. Without this window the gate's
			// removal is invisible — abort would simply win the race.
			require.Never(t, func() bool { return len(bucket.Objects()) > 0 },
				time.Second, 10*time.Millisecond,
				"block persisted while transaction open")

			if committed {
				require.NoError(t, tx.Commit())
			} else {
				tx.Abort()
			}
			wg.Wait() // block persisted (or skipped if emptied)

			want := int64(3)
			if committed {
				want = 6
			}
			require.Equal(t, want, countRows(t, db, "test"))

			require.NoError(t, c.Close())
			c, db = open()
			t.Cleanup(func() { require.NoError(t, c.Close()) })
			require.Equal(t, want, countRows(t, db, "test"))
		})
	}
}

// Test_Transaction_ConcurrentWrites races transactions (including aborts,
// which unlink LSM parts) against plain single-table writes. Run with
// -race; the invariant is that non-transactional rows always survive and
// aborted rows never do.
func Test_Transaction_ConcurrentWrites(t *testing.T) {
	c, err := New(
		WithLogger(newTestLogger(t)),
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, c.Close()) })
	db, err := c.DB(context.Background(), "test")
	require.NoError(t, err)
	table, err := db.Table("test", NewTableConfig(dynparquet.SampleDefinition()))
	require.NoError(t, err)

	r := testSamplesRecord(t, 3)
	defer r.Release()

	const iterations = 50
	ctx := context.Background()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			if _, err := table.InsertRecord(ctx, r); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			tx := db.Begin()
			tbl, err := tx.GetTable("test")
			if err != nil {
				t.Error(err)
				return
			}
			if err := tbl.InsertRecord(ctx, r); err != nil {
				t.Error(err)
				return
			}
			if i%2 == 0 {
				tx.Abort()
			} else if err := tx.Commit(); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	wg.Wait()

	// 50 plain writes + 25 committed transactions, 3 rows each.
	require.Equal(t, int64((iterations+iterations/2)*3), countRows(t, db, "test"))
}

// Test_Transaction_CompactionDuringOpenTx pins the H1 regression from the
// adversarial review: L0 compaction triggered while a transaction is open
// (and nothing in L0 is at or below the watermark) must NOT carry the
// transaction's parts into L1 — Abort can only remove from L0, so a
// compacted part would resurrect aborted rows and double-release against
// Remove.
func Test_Transaction_CompactionDuringOpenTx(t *testing.T) {
	c, err := New(
		WithLogger(newTestLogger(t)),
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, c.Close()) })
	db, err := c.DB(context.Background(), "test")
	require.NoError(t, err)
	table, err := db.Table("test", NewTableConfig(dynparquet.SampleDefinition()))
	require.NoError(t, err)

	r := testSamplesRecord(t, 3)
	defer r.Release()

	ctx := context.Background()
	tx := db.Begin()
	tbl, err := tx.GetTable("test")
	require.NoError(t, err)
	require.NoError(t, tbl.InsertRecord(ctx, r))

	// Force a full compaction pass while the transaction is open. The
	// open transaction's parts are the only L0 content and are above the
	// watermark; the merge must leave them alone.
	require.NoError(t, table.EnsureCompaction())

	tx.Abort()
	require.Equal(t, int64(0), countRows(t, db, "test"))

	// The watermark advanced and the table still works.
	_, err = table.InsertRecord(ctx, r)
	require.NoError(t, err)
	require.Equal(t, int64(3), countRows(t, db, "test"))
}

// Test_Transaction_DiscardRotationDuringOpenTx pins the M1 regression from
// the adversarial review: a skipPersist (discard) rotation while a
// transaction is open must not race the transaction's Abort —
// dropPendingBlock releases the index's parts, and an Abort landing after
// that would release the transaction's parts a second time. The watermark
// gate in writeBlock holds the drop until the transaction finishes.
func Test_Transaction_DiscardRotationDuringOpenTx(t *testing.T) {
	c, err := New(
		WithLogger(newTestLogger(t)),
		WithManualBlockRotation(),
	)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, c.Close()) })
	db, err := c.DB(context.Background(), "test")
	require.NoError(t, err)
	table, err := db.Table("test", NewTableConfig(dynparquet.SampleDefinition()))
	require.NoError(t, err)

	r := testSamplesRecord(t, 3)
	defer r.Release()

	ctx := context.Background()
	tx := db.Begin()
	tbl, err := tx.GetTable("test")
	require.NoError(t, err)
	require.NoError(t, tbl.InsertRecord(ctx, r))

	var wg sync.WaitGroup
	wg.Add(1)
	require.NoError(t, table.RotateBlock(ctx, table.ActiveBlock(),
		WithRotateBlockSkipPersist(), WithRotateBlockWaitGroup(&wg)))

	// The drop must not proceed while the transaction is open.
	dropped := make(chan struct{})
	go func() { wg.Wait(); close(dropped) }()
	select {
	case <-dropped:
		t.Fatal("discard rotation dropped the block while the transaction was open")
	case <-time.After(500 * time.Millisecond):
	}

	tx.Abort() // single release of the tx parts; the drop then releases the rest
	<-dropped

	// The discard dropped the whole block; nothing is visible and the
	// table still works (no refcount underflow under -race/assert).
	require.Equal(t, int64(0), countRows(t, db, "test"))
	_, err = table.InsertRecord(ctx, r)
	require.NoError(t, err)
	require.Equal(t, int64(3), countRows(t, db, "test"))
}
