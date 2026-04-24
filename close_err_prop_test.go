package frostdb

import (
	"context"
	"errors"
	"io"
	"sync/atomic"
	"testing"

	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/require"
	"github.com/thanos-io/objstore"

	"github.com/polarsignals/frostdb/dynparquet"
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
