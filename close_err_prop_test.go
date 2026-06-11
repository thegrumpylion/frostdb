package frostdb

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/require"
	"github.com/thanos-io/objstore"

	"github.com/polarsignals/frostdb/dynparquet"
	"github.com/polarsignals/frostdb/query/logicalplan"
)

// failingUploadSink wraps a normal objstore.Bucket-backed DataSink
// but forces Upload to return a sentinel error. Scan/Prefixes/Delete
// stay functional so DB recovery and bucket listing work.
//
// Used by TestCloseSurfacesUploadError to demonstrate that
// DB.Close / ColumnStore.Close currently swallows errors from the
// final shutdown-rotation's block.Persist → DataSink.Upload path.
type failingUploadSink struct {
	*DefaultObjstoreBucket
	err         error
	uploadCalls atomic.Int64
}

func (f *failingUploadSink) Upload(ctx context.Context, name string, r io.Reader) error {
	f.uploadCalls.Add(1)
	// Drain the pipe so the goroutine inside block.Persist that's
	// writing into it doesn't block forever. The error is returned
	// AFTER the drain.
	_, _ = io.Copy(io.Discard, r)
	return f.err
}

// closeErrPropRow is a minimal schema for the test. Any shape that
// produces a non-trivial block works — the test doesn't care what's
// inside, only that a block rotation runs at Close with a DataSink
// attached.
type closeErrPropRow struct {
	ID    uint64 `frostdb:"id,asc(0)"`
	Value uint64 `frostdb:"value"`
}

// TestCloseSurfacesUploadError pins the fix for observer's
// store-followups #3: when the shutdown-triggered block rotation
// fails at Upload, the error must propagate through
// ColumnStore.Close so callers can detect the data-loss window.
//
// Pre-fix trace:
//   DB.Close (db.go:983)
//     → table.writeBlock(block, tx, false)   -- return value ignored
//         → block.Persist()
//             → sink.Upload(...)  -- returns sentinelUploadErr
//         ← error logged via level.Error, return (writeBlock has
//                                                  no error return)
//       ↑ ERROR LOST HERE
//     → closeInternal() → nil (WAL already drained)
//   ← DB.Close returns nil
//   ColumnStore.Close collects DB.Close returns via errgroup → nil
//
// The test detects this by: (a) confirming Upload was called (so
// the rotation happened), (b) asserting ColumnStore.Close returned
// an error wrapping the sentinel. Pre-fix (a) passes and (b) fails.
// Post-fix both pass.
func TestCloseSurfacesUploadError(t *testing.T) {
	ctx := context.Background()
	logger := newTestLogger(t)

	sentinelUploadErr := errors.New("sentinel: upload failed at seal")
	baseBucket := objstore.NewInMemBucket()
	sink := &failingUploadSink{
		DefaultObjstoreBucket: NewDefaultObjstoreBucket(baseBucket),
		err:                   sentinelUploadErr,
	}

	dir := t.TempDir()
	c, err := New(
		WithLogger(logger),
		WithWAL(),
		WithStoragePath(dir),
		WithReadWriteStorage(sink),
	)
	require.NoError(t, err)

	db, err := c.DB(ctx, "test")
	require.NoError(t, err)

	table, err := NewGenericTable[closeErrPropRow](db, "closeerr", memory.NewGoAllocator())
	require.NoError(t, err)

	_, err = table.Write(ctx,
		closeErrPropRow{ID: 1, Value: 10},
		closeErrPropRow{ID: 2, Value: 20},
	)
	require.NoError(t, err)

	// Close triggers DB.Close → table.writeBlock → block.Persist →
	// sink.Upload, which returns sentinelUploadErr.
	closeErr := c.Close()

	// Preconditions: the shutdown rotation actually attempted the
	// upload. If Upload wasn't called, the test is misdiagnosing —
	// the rotation path was skipped for some unrelated reason.
	require.Greater(t, sink.uploadCalls.Load(), int64(0),
		"test setup broken: Upload never called, so error propagation is moot")

	// The actual contract: the Upload error must surface.
	require.Error(t, closeErr,
		"ColumnStore.Close returned nil despite DataSink.Upload failing during shutdown rotation — the error was silently dropped somewhere in DB.Close/writeBlock/Persist")
	require.ErrorIs(t, closeErr, sentinelUploadErr,
		"ColumnStore.Close surfaced an error but did not wrap the sentinel: %v", closeErr)
}

// TestCloseFailedUploadPreservesWALForRecovery is the load-bearing
// correctness test for the fix. Proves the end-to-end invariant the
// fix actually protects: when Upload fails during the shutdown
// rotation, the records that didn't make it to the bucket must
// survive via the WAL so a subsequent process lifetime can recover
// them.
//
// Pre-fix (upstream main), DB.Close proceeds past the silently-
// dropped writeBlock error and runs dropStorage, which deletes the
// storagePath directory — including the WAL. The records are
// permanently lost: no bucket entry (Upload failed), no WAL entry
// (dropStorage deleted it). This test fails at the
// "survived = N" assertion when run against pre-fix code.
//
// Post-fix, writeBlock surfaces the error through Close, which
// returns early before dropStorage. The WAL stays on disk. The
// second process lifetime replays it via DB.Open, re-ingests the
// records, and the final Scan counts them. This test passes.
//
// Two lifetimes share:
//   - the storagePath (WAL + snapshots lives there)
//   - the bucket (empty after lifetime 1 because Upload always
//     failed, so nothing was persisted)
//
// Lifetime 2 uses a WORKING sink (not the failing one) to
// demonstrate that recovery also successfully re-persists on the
// next rotation cycle.
func TestCloseFailedUploadPreservesWALForRecovery(t *testing.T) {
	ctx := context.Background()
	logger := newTestLogger(t)

	sentinelUploadErr := errors.New("sentinel: upload failed at seal")
	bucket := objstore.NewInMemBucket()
	dir := t.TempDir()

	const numRows = 5
	rowsIn := make([]closeErrPropRow, numRows)
	for i := 0; i < numRows; i++ {
		rowsIn[i] = closeErrPropRow{ID: uint64(i + 1), Value: uint64((i + 1) * 10)}
	}

	// --- Lifetime 1: write + Close with failing Upload ---
	{
		sink := &failingUploadSink{
			DefaultObjstoreBucket: NewDefaultObjstoreBucket(bucket),
			err:                   sentinelUploadErr,
		}

		c, err := New(
			WithLogger(logger),
			WithWAL(),
			WithStoragePath(dir),
			WithReadWriteStorage(sink),
		)
		require.NoError(t, err)

		db, err := c.DB(ctx, "test")
		require.NoError(t, err)

		table, err := NewGenericTable[closeErrPropRow](db, "recovery", memory.NewGoAllocator())
		require.NoError(t, err)

		_, err = table.Write(ctx, rowsIn...)
		require.NoError(t, err)

		closeErr := c.Close()
		// Log but don't require — the error-propagation contract is
		// tested by TestCloseSurfacesUploadError. This test's
		// load-bearing assertion is data survival across the
		// failed-Upload boundary, which pre-fix fails regardless of
		// whether Close returned an error (the WAL gets deleted
		// either way).
		t.Logf("lifetime 1 Close returned: %v", closeErr)

		// Upload was attempted — if not, the test setup is broken
		// and recovery claims are moot.
		require.Greater(t, sink.uploadCalls.Load(), int64(0), "test setup: Upload never called")
	}

	// --- Lifetime 2: reopen with a working sink, verify WAL replay ---
	//
	// Fresh ColumnStore, fresh sink (no failure injection). The
	// storagePath from lifetime 1 is still on disk. DB.Open reads it,
	// replays the WAL into the table, and the rows appear via Scan.
	{
		workingSink := NewDefaultObjstoreBucket(bucket)
		c, err := New(
			WithLogger(logger),
			WithWAL(),
			WithStoragePath(dir),
			WithReadWriteStorage(workingSink),
		)
		require.NoError(t, err)
		defer c.Close()

		db, err := c.DB(ctx, "test")
		require.NoError(t, err)

		// Re-register the generic table to bind the schema. NewGenericTable
		// is idempotent on an already-existing FrostDB table.
		table, err := NewGenericTable[closeErrPropRow](db, "recovery", memory.NewGoAllocator())
		require.NoError(t, err)
		defer table.Release()

		// Scan the recovered table.
		pool := memory.NewCheckedAllocator(memory.DefaultAllocator)
		defer pool.AssertSize(t, 0)
		survived := int64(0)
		err = table.View(ctx, func(ctx context.Context, tx uint64) error {
			return table.Iterator(
				ctx,
				tx,
				pool,
				[]logicalplan.Callback{func(_ context.Context, ar arrow.Record) error {
					survived += ar.NumRows()
					return nil
				}},
			)
		})
		require.NoError(t, err)

		// The load-bearing assertion: records must survive the
		// failed-Upload Close via WAL replay. Pre-fix this reads 0;
		// post-fix it reads numRows.
		require.Equal(t, int64(numRows), survived,
			"WAL replay did not recover records after failed-Upload Close — dropStorage destroyed the WAL even though Persist failed (pre-fix data-loss window)")
	}
}

// TestCloseCleanPathReturnsNil is the positive control: with a
// well-behaved sink (no Upload errors), Close must return nil.
// Guards against an over-correction in the fix that would make
// Close fail spuriously.
func TestCloseCleanPathReturnsNil(t *testing.T) {
	ctx := context.Background()
	logger := newTestLogger(t)

	bucket := objstore.NewInMemBucket()
	sink := NewDefaultObjstoreBucket(bucket)

	dir := t.TempDir()
	c, err := New(
		WithLogger(logger),
		WithWAL(),
		WithStoragePath(dir),
		WithReadWriteStorage(sink),
	)
	require.NoError(t, err)

	db, err := c.DB(ctx, "test")
	require.NoError(t, err)

	table, err := NewGenericTable[closeErrPropRow](db, "closeerr_clean", memory.NewGoAllocator())
	require.NoError(t, err)

	_, err = table.Write(ctx, closeErrPropRow{ID: 1, Value: 10})
	require.NoError(t, err)

	require.NoError(t, c.Close(), "clean Close path should return nil")

	// Sanity: something actually landed in the bucket.
	objects := 0
	require.NoError(t, bucket.Iter(ctx, "", func(string) error { objects++; return nil }, objstore.WithRecursiveIter))
	require.NotZero(t, objects, "clean Close should have uploaded at least one block")

	// Touch the unused import so go vet/compile is happy even if
	// dynparquet usage is removed later.
	_ = dynparquet.Schema{}
}
