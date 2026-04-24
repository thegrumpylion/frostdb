package frostdb

import (
	"context"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/stretchr/testify/require"
	"github.com/thanos-io/objstore"

	"github.com/polarsignals/frostdb/query"
	"github.com/polarsignals/frostdb/query/logicalplan"
)

// bucketsRow mirrors the shape Observer's histogram schema uses: a
// monotonically-increasing identifier plus a non-empty []uint64 list
// column. Any list-element type that record_builder can produce
// (int64, uint64, float64, bool, string) routes through the same
// writeList dispatch, so this shape is representative of the whole
// class of bugs.
type bucketsRow struct {
	ID      uint64   `frostdb:"id,asc(0)"`
	Buckets []uint64 `frostdb:"buckets"`
}

// scalarRow is the isolation control for bucketsRow — same outer
// shape, scalar Buckets field. Used to confirm the rotation-loss
// path only breaks on list element types, not on the whole
// rotate-on-close flow.
type scalarRow struct {
	ID    uint64 `frostdb:"id,asc(0)"`
	Value uint64 `frostdb:"value"`
}

// TestScalarColumnSurvivesRotateAndReopen is the control. If this
// passes and TestListColumnSurvivesRotateAndReopen fails, the bug
// is specifically in the list-element serialization, not in the
// broader rotate/persist path.
func TestScalarColumnSurvivesRotateAndReopen(t *testing.T) {
	dir := t.TempDir()
	bucket := objstore.NewInMemBucket()
	sink := NewDefaultObjstoreBucket(bucket)
	mem := memory.NewGoAllocator()
	ctx := context.Background()
	logger := newTestLogger(t)

	input := []scalarRow{{ID: 1, Value: 10}, {ID: 2, Value: 20}, {ID: 3, Value: 30}}

	c, err := New(WithLogger(logger), WithWAL(), WithStoragePath(dir), WithReadWriteStorage(sink))
	require.NoError(t, err)
	db, err := c.DB(ctx, "test")
	require.NoError(t, err)
	table, err := NewGenericTable[scalarRow](db, "scalars", mem)
	require.NoError(t, err)
	_, err = table.Write(ctx, input...)
	require.NoError(t, err)
	require.NoError(t, c.Close())

	objects := 0
	require.NoError(t, bucket.Iter(ctx, "", func(string) error { objects++; return nil }, objstore.WithRecursiveIter))
	require.NotZero(t, objects, "scalar column rotate-on-close produced no bucket objects")
}

// TestListColumnSurvivesRotateAndReopen writes rows containing a
// non-empty []uint64 list column to a ColumnStore backed by both a
// WAL and a DataSink bucket, forces a block rotation via Close, and
// then reopens the store and reads the column back. It asserts that
// the list column values survive the round-trip.
//
// Filed as Observer's workaround in
// docs/issues/frostdb-list-column-rotation-loss.md. The workaround
// reseeds histograms on every cache hit because bucket_counts reads
// back EMPTY after rotation. Root cause: pqarrow/parquet.go
// writeList has a type switch over list element types that omits
// *array.Uint64, returning an error from Serialize; that error
// unwinds through block.Persist() which deletes the partial upload,
// leaving the bucket empty. The upstream fix adds the missing case
// plus a generic fallback for remaining Arrow primitive types.
func TestListColumnSurvivesRotateAndReopen(t *testing.T) {
	dir := t.TempDir()
	bucket := objstore.NewInMemBucket()
	sink := NewDefaultObjstoreBucket(bucket)
	mem := memory.NewGoAllocator()
	ctx := context.Background()
	logger := newTestLogger(t)

	input := []bucketsRow{
		{ID: 1, Buckets: []uint64{10, 20, 30}},
		{ID: 2, Buckets: []uint64{40, 50}},
		{ID: 3, Buckets: []uint64{60, 70, 80, 90}},
	}

	// First open: write + rotate via Close. A DataSink is configured,
	// so DB.Close triggers a final writeBlock → block.Persist →
	// sink.Upload. Matches Observer's shutdown rotation path.
	{
		c, err := New(WithLogger(logger), WithWAL(), WithStoragePath(dir), WithReadWriteStorage(sink))
		require.NoError(t, err)
		db, err := c.DB(ctx, "test")
		require.NoError(t, err)
		table, err := NewGenericTable[bucketsRow](db, "buckets", mem)
		require.NoError(t, err)
		_, err = table.Write(ctx, input...)
		require.NoError(t, err)
		require.NoError(t, c.Close())
	}

	// Second open: fresh ColumnStore against the same storage + bucket.
	c, err := New(WithLogger(logger), WithWAL(), WithStoragePath(dir), WithReadWriteStorage(sink))
	require.NoError(t, err)
	defer c.Close()

	db, err := c.DB(ctx, "test")
	require.NoError(t, err)

	// NewGenericTable on an existing FrostDB table is idempotent —
	// the call registers the schema in this process.
	_, err = NewGenericTable[bucketsRow](db, "buckets", mem)
	require.NoError(t, err)

	// Scan the persisted data and collect (id, buckets) per row.
	engine := query.NewEngine(mem, db.TableProvider())
	var got []arrow.Record
	defer func() {
		for _, r := range got {
			r.Release()
		}
	}()
	err = engine.ScanTable("buckets").
		Project(logicalplan.Col("id"), logicalplan.Col("buckets")).
		Execute(ctx, func(_ context.Context, ar arrow.Record) error {
			ar.Retain()
			got = append(got, ar)
			return nil
		})
	require.NoError(t, err)

	seen := make(map[uint64][]uint64)
	for _, rec := range got {
		idCol := rec.Column(rec.Schema().FieldIndices("id")[0]).(*array.Uint64)
		bucketsCol := rec.Column(rec.Schema().FieldIndices("buckets")[0]).(*array.List)
		bucketValues := bucketsCol.ListValues().(*array.Uint64)

		for i := 0; i < int(rec.NumRows()); i++ {
			id := idCol.Value(i)
			if bucketsCol.IsNull(i) {
				seen[id] = nil
				continue
			}
			start, end := bucketsCol.ValueOffsets(i)
			out := make([]uint64, 0, end-start)
			for j := start; j < end; j++ {
				out = append(out, bucketValues.Value(int(j)))
			}
			seen[id] = out
		}
	}

	require.Len(t, seen, len(input), "row count lost across rotation")
	for _, want := range input {
		require.Equal(t, want.Buckets, seen[want.ID],
			"buckets for id=%d lost or mangled across rotation+reopen", want.ID)
	}
}
