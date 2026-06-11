package index

import (
	"context"
	"errors"
	"io"
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/parquet-go/parquet-go"
	"github.com/stretchr/testify/require"

	"github.com/polarsignals/frostdb/dynparquet"
	"github.com/polarsignals/frostdb/parts"
)

func compactParts(w io.Writer, compact []parts.Part, _ ...parquet.WriterOption) (int64, error) {
	schema := dynparquet.NewSampleSchema()
	bufs := []dynparquet.DynamicRowGroup{}
	var size int64
	for _, part := range compact {
		size += part.Size()
		buf, err := part.AsSerializedBuffer(schema)
		if err != nil {
			return 0, err
		}
		bufs = append(bufs, buf.MultiDynamicRowGroup())
	}
	merged, err := schema.MergeDynamicRowGroups(bufs)
	if err != nil {
		return 0, err
	}
	err = func() error {
		writer, err := schema.GetWriter(w, merged.DynamicColumns(), false)
		if err != nil {
			return err
		}
		defer writer.Close()

		rows := merged.Rows()
		defer rows.Close()

		buf := make([]parquet.Row, merged.NumRows())
		if _, err := rows.ReadRows(buf); err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		if _, err := writer.WriteRows(buf); err != nil && !errors.Is(err, io.EOF) {
			return err
		}

		return nil
	}()
	if err != nil {
		return 0, err
	}

	return size, nil
}

func check(t *testing.T, lsm *LSM, records, buffers int) {
	t.Helper()
	seen := map[SentinelType]bool{}
	lsm.partList.Iterate(func(node *Node) bool {
		if node.part == nil {
			if seen[node.sentinel] {
				t.Fatal("duplicate sentinel")
			}
			seen[node.sentinel] = true
		}
		return true
	})
	rec := 0
	buf := 0
	require.NoError(t, lsm.Scan(context.Background(), "", nil, nil, math.MaxUint64, func(_ context.Context, v any) error {
		switch v.(type) {
		case arrow.Record:
			rec++
		case dynparquet.DynamicRowGroup:
			buf++
		}
		return nil
	}))
	require.Equal(t, records, rec)
	require.Equal(t, buf, buffers)
}

func Test_LSM_Basic(t *testing.T) {
	t.Parallel()
	lsm, err := NewLSM("test", nil, []*LevelConfig{
		{Level: L0, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L1, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L2, MaxSize: 1024 * 1024 * 1024},
	},
		func() uint64 { return math.MaxUint64 },
	)
	require.NoError(t, err)

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	lsm.Add(1, r)
	lsm.Add(2, r)
	lsm.Add(3, r)
	check(t, lsm, 3, 0)
	require.NoError(t, lsm.merge(L0))
	check(t, lsm, 0, 1)
	lsm.Add(4, r)
	check(t, lsm, 1, 1)
	lsm.Add(5, r)
	check(t, lsm, 2, 1)
	require.NoError(t, lsm.merge(L0))
	check(t, lsm, 0, 2)
	lsm.Add(6, r)
	check(t, lsm, 1, 2)
	require.NoError(t, lsm.merge(L1))
	check(t, lsm, 1, 1)
	require.NoError(t, lsm.merge(L0))
	check(t, lsm, 0, 2)
}

func Test_LSM_DuplicateSentinel(t *testing.T) {
	t.Parallel()
	lsm, err := NewLSM("test", nil, []*LevelConfig{
		{Level: L0, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L1, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L2, MaxSize: 1024 * 1024 * 1024},
	},
		func() uint64 { return math.MaxUint64 },
	)
	require.NoError(t, err)

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	lsm.Add(1, r)
	lsm.Add(2, r)
	lsm.Add(3, r)
	check(t, lsm, 3, 0)
	require.NoError(t, lsm.merge(L0))
	check(t, lsm, 0, 1)
	require.NoError(t, lsm.merge(L0))
	check(t, lsm, 0, 1)
}

func Test_LSM_Compaction(t *testing.T) {
	t.Parallel()
	lsm, err := NewLSM("test", nil, []*LevelConfig{
		{Level: L0, MaxSize: 1, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L1, MaxSize: 1024 * 1024 * 1024},
	},
		func() uint64 { return math.MaxUint64 },
	)
	require.NoError(t, err)

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	lsm.Add(1, r)
	require.Eventually(t, func() bool {
		return lsm.sizes[L0].Load() == 0 && lsm.sizes[L1].Load() != 0
	}, 30*time.Second, 10*time.Millisecond)
}

func Test_LSM_CascadeCompaction(t *testing.T) {
	t.Parallel()
	lsm, err := NewLSM("test", nil, []*LevelConfig{
		{Level: L0, MaxSize: 257, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L1, MaxSize: 2281, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L2, MaxSize: 2281, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: 3, MaxSize: 2281, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: 4, MaxSize: 2281},
	},
		func() uint64 { return math.MaxUint64 },
	)
	require.NoError(t, err)

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	lsm.Add(1, r)
	require.Eventually(t, func() bool {
		return lsm.sizes[L0].Load() == 0 &&
			lsm.sizes[L1].Load() != 0 &&
			lsm.sizes[L2].Load() == 0 &&
			lsm.sizes[3].Load() == 0 &&
			lsm.sizes[4].Load() == 0
	}, 3*time.Second, 10*time.Millisecond)
	lsm.Add(2, r)
	require.Eventually(t, func() bool {
		return lsm.sizes[L0].Load() == 0 &&
			lsm.sizes[L1].Load() == 0 &&
			lsm.sizes[L2].Load() == 0 &&
			lsm.sizes[3].Load() == 0 &&
			lsm.sizes[4].Load() != 0
	}, 30*time.Second, 10*time.Millisecond)
}

func Test_LSM_InOrderInsert(t *testing.T) {
	t.Parallel()
	lsm, err := NewLSM("test", nil, []*LevelConfig{
		{Level: L0, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L1, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L2, MaxSize: 1024 * 1024 * 1024},
	},
		func() uint64 { return math.MaxUint64 },
	)
	require.NoError(t, err)

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	wg := &sync.WaitGroup{}
	workers := 100
	inserts := 100
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < inserts; j++ {
				lsm.Add(rand.Uint64(), r)
			}
		}()
	}
	wg.Wait()

	tx := make([]uint64, 0, workers*inserts)
	lsm.Iterate(func(node *Node) bool {
		if node.part != nil {
			tx = append(tx, node.part.TX())
		}
		return true
	})

	// check that the transactions are sorted in descending order
	require.True(t, slices.IsSortedFunc[[]uint64, uint64](tx, func(i, j uint64) int {
		if i < j {
			return 1
		} else if i > j {
			return -1
		}

		return 0
	}))
}

// newRemoveTestLSM builds a 3-level LSM whose watermark is pinned to the
// given function, for Remove tests.
func newRemoveTestLSM(t *testing.T, watermark func() uint64) *LSM {
	t.Helper()
	lsm, err := NewLSM("test", nil, []*LevelConfig{
		{Level: L0, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L1, MaxSize: 1024 * 1024 * 1024, Type: CompactionTypeParquetMemory, Compact: compactParts},
		{Level: L2, MaxSize: 1024 * 1024 * 1024},
	}, watermark)
	require.NoError(t, err)
	return lsm
}

// txsInL0 returns the part transaction ids currently linked in the list,
// front to back.
func txsInL0(lsm *LSM) []uint64 {
	txs := []uint64{}
	lsm.Iterate(func(node *Node) bool {
		if node.part != nil {
			txs = append(txs, node.part.TX())
		}
		return true
	})
	return txs
}

func Test_LSM_Remove(t *testing.T) {
	t.Parallel()

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	t.Run("adjacent same-tx parts", func(t *testing.T) {
		lsm := newRemoveTestLSM(t, func() uint64 { return 0 })
		lsm.Add(1, r)
		lsm.Add(2, r)
		lsm.Add(2, r)
		lsm.Add(2, r)
		lsm.Add(3, r)

		lsm.Remove(2)
		require.Equal(t, []uint64{3, 1}, txsInL0(lsm))
	})

	t.Run("first node match", func(t *testing.T) {
		lsm := newRemoveTestLSM(t, func() uint64 { return 0 })
		lsm.Add(1, r)
		lsm.Add(2, r)
		// tx 2 sits at the front of the list (descending order).
		lsm.Remove(2)
		require.Equal(t, []uint64{1}, txsInL0(lsm))
	})

	t.Run("remove all parts", func(t *testing.T) {
		lsm := newRemoveTestLSM(t, func() uint64 { return 0 })
		lsm.Add(7, r)
		lsm.Add(7, r)
		lsm.Remove(7)
		require.Empty(t, txsInL0(lsm))
		require.Zero(t, lsm.LevelSize(L0))
	})

	t.Run("remove nonexistent tx is a no-op", func(t *testing.T) {
		lsm := newRemoveTestLSM(t, func() uint64 { return 0 })
		lsm.Add(1, r)
		size := lsm.LevelSize(L0)
		lsm.Remove(42)
		require.Equal(t, []uint64{1}, txsInL0(lsm))
		require.Equal(t, size, lsm.LevelSize(L0))
	})

	t.Run("size accounting", func(t *testing.T) {
		lsm := newRemoveTestLSM(t, func() uint64 { return 0 })
		lsm.Add(1, r)
		sizeOne := lsm.LevelSize(L0)
		lsm.Add(2, r)
		lsm.Add(2, r)
		lsm.Remove(2)
		require.Equal(t, sizeOne, lsm.LevelSize(L0))
	})

	t.Run("does not cross into compacted levels", func(t *testing.T) {
		// A part with the same tx below the L0 sentinel must survive:
		// Remove only scans L0. (Reachable only for completed txns, but
		// the scan bound is part of the contract.)
		lsm := newRemoveTestLSM(t, func() uint64 { return math.MaxUint64 })
		lsm.Add(1, r)
		lsm.Add(2, r)
		require.NoError(t, lsm.merge(L0)) // 1 and 2 now live in L1
		lsm.Add(2, r)                     // a fresh L0 part with tx 2
		lsm.Remove(2)

		count := 0
		lsm.Iterate(func(node *Node) bool {
			if node.part != nil {
				count++
			}
			return true
		})
		require.Equal(t, 1, count) // only the compacted L1 part remains
	})

	t.Run("concurrent add and remove", func(t *testing.T) {
		lsm := newRemoveTestLSM(t, func() uint64 { return 0 })
		const iterations = 200
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				lsm.Add(uint64(1000+i), r)
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tx := uint64(2_000_000 + i)
				lsm.Add(tx, r)
				lsm.Remove(tx)
			}
		}()
		wg.Wait()

		txs := txsInL0(lsm)
		require.Len(t, txs, iterations) // every kept part survived
		for _, tx := range txs {
			require.Less(t, tx, uint64(2_000_000)) // every removed part is gone
		}
	})
}

// Test_LSM_MergeRespectsWatermark pins the L0 compaction gate: when NO L0
// part is at or below the watermark, merge must be a no-op. The previous
// fallback silently compacted the WHOLE L0 list — carrying an open
// transaction's parts into L1, beyond Remove's reach, resurrecting aborted
// rows and setting up a double release against a concurrent Remove.
func Test_LSM_MergeRespectsWatermark(t *testing.T) {
	t.Parallel()

	samples := dynparquet.NewTestSamples()
	r, err := samples.ToRecord()
	require.NoError(t, err)

	lsm := newRemoveTestLSM(t, func() uint64 { return 0 }) // nothing is ever committed
	lsm.Add(5, r)
	lsm.Add(6, r)
	require.NoError(t, lsm.merge(L0))

	require.Equal(t, []uint64{6, 5}, txsInL0(lsm)) // still in L0, uncompacted
	require.Zero(t, lsm.LevelSize(L1))

	// And Remove can still reach them, as the abort path requires.
	lsm.Remove(5)
	require.Equal(t, []uint64{6}, txsInL0(lsm))
}
