package expr

import (
	"testing"

	"github.com/parquet-go/parquet-go"
	"github.com/stretchr/testify/require"

	"github.com/polarsignals/frostdb/query/logicalplan"
)

type FakeColumnChunk struct {
	index     parquet.ColumnIndex
	numValues int64
}

func (f *FakeColumnChunk) Type() parquet.Type                        { return nil }
func (f *FakeColumnChunk) Column() int                               { return 0 }
func (f *FakeColumnChunk) Pages() parquet.Pages                      { return nil }
func (f *FakeColumnChunk) ColumnIndex() (parquet.ColumnIndex, error) { return f.index, nil }
func (f *FakeColumnChunk) OffsetIndex() (parquet.OffsetIndex, error) { return nil, nil }
func (f *FakeColumnChunk) BloomFilter() parquet.BloomFilter          { return nil }
func (f *FakeColumnChunk) NumValues() int64                          { return f.numValues }

type FakeColumnIndex struct {
	numPages  int
	min       parquet.Value
	max       parquet.Value
	nullCount int64
}

func (f *FakeColumnIndex) NumPages() int {
	return f.numPages
}
func (f *FakeColumnIndex) NullCount(int) int64        { return f.nullCount }
func (f *FakeColumnIndex) NullPage(int) bool          { return false }
func (f *FakeColumnIndex) MinValue(int) parquet.Value { return f.min }
func (f *FakeColumnIndex) MaxValue(int) parquet.Value { return f.max }
func (f *FakeColumnIndex) IsAscending() bool          { return false }
func (f *FakeColumnIndex) IsDescending() bool         { return false }

// FakePagesColumnIndex is a per-page fake: page i reports mins[i]/maxs[i].
// A null parquet.Value in those slices models an all-null page, matching
// parquet-go's fileColumnIndex, which returns Value{} for null pages.
type FakePagesColumnIndex struct {
	mins, maxs []parquet.Value
	nullCounts []int64
}

func (f *FakePagesColumnIndex) NumPages() int             { return len(f.mins) }
func (f *FakePagesColumnIndex) NullCount(i int) int64     { return f.nullCounts[i] }
func (f *FakePagesColumnIndex) NullPage(i int) bool       { return f.mins[i].IsNull() }
func (f *FakePagesColumnIndex) MinValue(i int) parquet.Value { return f.mins[i] }
func (f *FakePagesColumnIndex) MaxValue(i int) parquet.Value { return f.maxs[i] }
func (f *FakePagesColumnIndex) IsAscending() bool         { return false }
func (f *FakePagesColumnIndex) IsDescending() bool        { return false }

// requireFaithful rejects fixtures real files cannot produce:
// fileColumnIndex gates MinValue and MaxValue on the same null_pages
// entry, so per-page min/max null-ness always agrees.
func requireFaithful(t *testing.T, f *FakePagesColumnIndex) {
	t.Helper()
	for i := range f.mins {
		require.Equal(t, f.mins[i].IsNull(), f.maxs[i].IsNull(),
			"page %d: min/max null-ness must agree", i)
	}
}

// Regression: an all-null page between value pages must not reset the
// accumulated min/max. Observed in production (observer L/XL datasets):
// sparse optional columns serialize as value pages interleaved with
// all-null pages; the old loop fed the null page's Value{} to compare,
// which treats it as ""/0, clobbering the accumulator and dropping every
// bound seen before the null page. Equality pushdown then pruned row
// groups that contained the value — silent row loss.
func Test_MinMax_NullPageBetweenValuePages(t *testing.T) {
	null := parquet.ValueOf(nil)

	// String chunk: true min "wf-00000" on page 0, all-null page 1,
	// larger min on page 2.
	strIndex := &FakePagesColumnIndex{
		mins:       []parquet.Value{parquet.ValueOf("wf-00000"), null, parquet.ValueOf("wf-00027")},
		maxs:       []parquet.Value{parquet.ValueOf("wf-07666"), null, parquet.ValueOf("wf-17464")},
		nullCounts: []int64{100, 1000, 100},
	}
	requireFaithful(t, strIndex)
	require.Equal(t, "wf-00000", Min(strIndex).String())
	require.Equal(t, "wf-17464", Max(strIndex).String())

	// Negative ints: null page's Value{} reads as 0 through Int64Type
	// comparison, so the symmetric Max bug needs max < 0 to show.
	intIndex := &FakePagesColumnIndex{
		mins:       []parquet.Value{parquet.ValueOf(int64(-50)), null, parquet.ValueOf(int64(-40))},
		maxs:       []parquet.Value{parquet.ValueOf(int64(-5)), null, parquet.ValueOf(int64(-10))},
		nullCounts: []int64{100, 1000, 100},
	}
	requireFaithful(t, intIndex)
	require.Equal(t, int64(-50), Min(intIndex).Int64())
	require.Equal(t, int64(-5), Max(intIndex).Int64())

	// End-to-end: equality on the true min must keep the chunk.
	// NumValues counts nulls too, so it must be >= the 2200 nulls
	// above for the fixture to be a physically possible chunk.
	chunk := &FakeColumnChunk{
		index:     strIndex,
		numValues: 2500,
	}
	satisfies, err := BinaryScalarOperation(chunk, parquet.ValueOf("wf-00000"), logicalplan.OpEq)
	require.NoError(t, err)
	require.True(t, satisfies)

	// Range form of the same loss: <= true min.
	satisfies, err = BinaryScalarOperation(chunk, parquet.ValueOf("wf-00000"), logicalplan.OpLtEq)
	require.NoError(t, err)
	require.True(t, satisfies)
}

// This is a regression test that ensures the Min/Max functions return a null
// value (instead of panicing) should they be passed a column chunk that only
// has null values.
func Test_MinMax_EmptyColumnChunk(t *testing.T) {
	fakeIndex := &FakeColumnIndex{
		numPages: 10,
	}

	v := Min(fakeIndex)
	require.True(t, v.IsNull())

	v = Max(fakeIndex)
	require.True(t, v.IsNull())
}

func TestBinaryScalarOperation(t *testing.T) {
	const numValues = 10
	for _, tc := range []struct {
		name string
		min  int
		max  int
		// -1 is interpreted as a null value.
		right     int
		nullCount int64
		op        logicalplan.Op
		// expectSatisfies is true if the predicate should be satisfied by the
		// column chunk.
		expectSatisfies bool
	}{
		{
			name:            "OpEqValueContained",
			min:             1,
			max:             10,
			right:           5,
			op:              logicalplan.OpEq,
			expectSatisfies: true,
		},
		{
			name:            "OpEqValueGt",
			min:             1,
			max:             10,
			right:           11,
			op:              logicalplan.OpEq,
			expectSatisfies: false,
		},
		{
			name:            "OpEqValueLt",
			min:             1,
			max:             10,
			right:           0,
			op:              logicalplan.OpEq,
			expectSatisfies: false,
		},
		{
			name:            "OpEqMaxBound",
			min:             1,
			max:             10,
			right:           10,
			op:              logicalplan.OpEq,
			expectSatisfies: true,
		},
		{
			name:            "OpEqMinBound",
			min:             1,
			max:             10,
			right:           1,
			op:              logicalplan.OpEq,
			expectSatisfies: true,
		},
		{
			name:            "OpEqNullValueNoMatch",
			right:           -1,
			nullCount:       0,
			op:              logicalplan.OpEq,
			expectSatisfies: false,
		},
		{
			name:            "OpEqNullValueMatch",
			right:           -1,
			nullCount:       1,
			op:              logicalplan.OpEq,
			expectSatisfies: true,
		},
		{
			name:            "OpEqNullColumn",
			right:           1,
			nullCount:       1,
			op:              logicalplan.OpEq,
			expectSatisfies: false,
		},
		{
			name:            "OpEqFullNullColumn",
			right:           1,
			nullCount:       10,
			op:              logicalplan.OpEq,
			expectSatisfies: false,
		},
		{
			name:            "OpGtFullNullColumn",
			right:           1,
			nullCount:       10,
			op:              logicalplan.OpGt,
			expectSatisfies: false,
		},
		{
			name:            "OpGtNullValueMatch",
			right:           -1,
			nullCount:       0,
			op:              logicalplan.OpGt,
			expectSatisfies: true,
		},
		{
			name:      "OpGtNullValueNoMatch",
			right:     -1,
			nullCount: 1,
			op:        logicalplan.OpGt,
			// expectSatisfies should probably be false once we optimize this.
			expectSatisfies: true,
		},
		{
			name:            "OpGtWithSomeNullValuesNoMatch",
			min:             1,
			max:             10,
			right:           11,
			nullCount:       1,
			op:              logicalplan.OpGt,
			expectSatisfies: false,
		},
		{
			name:            "OpGtWithSomeNullValuesMatch",
			min:             1,
			max:             10,
			right:           5,
			nullCount:       1,
			op:              logicalplan.OpGt,
			expectSatisfies: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.op == logicalplan.OpUnknown {
				t.Fatal("test programming error: remember to set operator")
			}
			minV := parquet.ValueOf(tc.min)
			maxV := parquet.ValueOf(tc.max)
			if tc.nullCount == numValues {
				// All values in page are null. Parquet doesn't have
				// well-defined min/max values in this case, but setting them
				// explicitly to null here will tickle some edge cases.
				minV = parquet.ValueOf(nil)
				maxV = parquet.ValueOf(nil)
			}
			fakeChunk := &FakeColumnChunk{
				index: &FakeColumnIndex{
					numPages:  1,
					min:       minV,
					max:       maxV,
					nullCount: tc.nullCount,
				},
				numValues: numValues,
			}
			var v parquet.Value
			if tc.right == -1 {
				v = parquet.ValueOf(nil)
			} else {
				v = parquet.ValueOf(tc.right)
			}
			res, err := BinaryScalarOperation(fakeChunk, v, tc.op)
			require.NoError(t, err)
			require.Equal(t, tc.expectSatisfies, res)
		})
	}
}
