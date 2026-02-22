package gorgonia

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

// makeF32Backing returns a float32 slice of the given size filled with 1.0.
func makeF32Backing(size int) []float32 {
	b := make([]float32, size)
	for i := range b {
		b[i] = 1.0
	}
	return b
}

// makeF32BackingVal returns a float32 slice of the given size filled with val.
func makeF32BackingVal(size int, val float32) []float32 {
	b := make([]float32, size)
	for i := range b {
		b[i] = val
	}
	return b
}

// --- Repeat Benchmarks ---

// BenchmarkRepeat_F32_Vector benchmarks repeating a float32 vector along axis 0.
func BenchmarkRepeat_F32_Vector(b *testing.B) {
	sizes := []int{1000, 10000, 100000}
	reps := []int{2, 4, 8}

	for _, size := range sizes {
		for _, rep := range reps {
			name := fmt.Sprintf("size=%d/rep=%d", size, rep)
			b.Run(name, func(b *testing.B) {
				backing := makeF32Backing(size)
				aT := tensor.New(tensor.WithShape(size), tensor.WithBacking(backing))

				g := NewGraph()
				a := NodeFromAny(g, aT, WithName("a"))
				repOp := newRepeatOp(0, a)
				repVal := NewI(rep)

				b.ReportAllocs()
				b.SetBytes(int64(size * rep * 4)) // float32 = 4 bytes
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if _, err := repOp.Do(aT, repVal); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkRepeat_F32_Matrix_Axis0 benchmarks repeating a float32 NxN matrix along axis 0.
func BenchmarkRepeat_F32_Matrix_Axis0(b *testing.B) {
	dims := []int{32, 128, 512}

	for _, n := range dims {
		name := fmt.Sprintf("N=%d", n)
		b.Run(name, func(b *testing.B) {
			backing := makeF32Backing(n * n)
			aT := tensor.New(tensor.WithShape(n, n), tensor.WithBacking(backing))

			g := NewGraph()
			a := NodeFromAny(g, aT, WithName("a"))
			repOp := newRepeatOp(0, a)
			repVal := NewI(2)

			b.ReportAllocs()
			b.SetBytes(int64(n * n * 2 * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := repOp.Do(aT, repVal); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkRepeat_F32_Matrix_Axis1 benchmarks repeating a float32 NxN matrix along axis 1.
func BenchmarkRepeat_F32_Matrix_Axis1(b *testing.B) {
	dims := []int{32, 128, 512}

	for _, n := range dims {
		name := fmt.Sprintf("N=%d", n)
		b.Run(name, func(b *testing.B) {
			backing := makeF32Backing(n * n)
			aT := tensor.New(tensor.WithShape(n, n), tensor.WithBacking(backing))

			g := NewGraph()
			a := NodeFromAny(g, aT, WithName("a"))
			repOp := newRepeatOp(1, a)
			repVal := NewI(2)

			b.ReportAllocs()
			b.SetBytes(int64(n * n * 2 * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := repOp.Do(aT, repVal); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// --- Broadcast Benchmarks (through TapeMachine) ---

// BenchmarkBroadcastAdd_F32 benchmarks BroadcastAdd of (N,1) + (1,N) via TapeMachine.
func BenchmarkBroadcastAdd_F32(b *testing.B) {
	dims := []int{32, 128, 512, 1024}

	for _, n := range dims {
		name := fmt.Sprintf("N=%d", n)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * n * 4))

			aBack := makeF32Backing(n)
			bBack := makeF32Backing(n)
			aT := tensor.New(tensor.WithShape(n, 1), tensor.WithBacking(aBack))
			bT := tensor.New(tensor.WithShape(1, n), tensor.WithBacking(bBack))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g := NewGraph()
				a := NodeFromAny(g, aT, WithName("a"))
				bN := NodeFromAny(g, bT, WithName("b"))

				c, err := BroadcastAdd(a, bN, []byte{1}, []byte{0})
				if err != nil {
					b.Fatal(err)
				}
				_ = c

				machine := NewTapeMachine(g)
				if err = machine.RunAll(); err != nil {
					b.Fatal(err)
				}
				machine.Close()
			}
		})
	}
}

// BenchmarkBroadcastMul_F32 benchmarks BroadcastHadamardProd of (N,1) * (1,N) via TapeMachine.
func BenchmarkBroadcastMul_F32(b *testing.B) {
	dims := []int{32, 128, 512, 1024}

	for _, n := range dims {
		name := fmt.Sprintf("N=%d", n)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * n * 4))

			aBack := makeF32BackingVal(n, 2.0)
			bBack := makeF32BackingVal(n, 3.0)
			aT := tensor.New(tensor.WithShape(n, 1), tensor.WithBacking(aBack))
			bT := tensor.New(tensor.WithShape(1, n), tensor.WithBacking(bBack))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g := NewGraph()
				a := NodeFromAny(g, aT, WithName("a"))
				bN := NodeFromAny(g, bT, WithName("b"))

				c, err := BroadcastHadamardProd(a, bN, []byte{1}, []byte{0})
				if err != nil {
					b.Fatal(err)
				}
				_ = c

				machine := NewTapeMachine(g)
				if err = machine.RunAll(); err != nil {
					b.Fatal(err)
				}
				machine.Close()
			}
		})
	}
}

// BenchmarkBroadcastAdd_RowVec benchmarks adding (N,M) + (1,M) row vector broadcast.
func BenchmarkBroadcastAdd_RowVec(b *testing.B) {
	cases := []struct {
		n, m int
	}{
		{32, 64},
		{128, 256},
		{512, 512},
	}

	for _, tc := range cases {
		name := fmt.Sprintf("N=%d_M=%d", tc.n, tc.m)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(tc.n * tc.m * 4))

			matBack := makeF32Backing(tc.n * tc.m)
			rowBack := makeF32BackingVal(tc.m, 10.0)
			matT := tensor.New(tensor.WithShape(tc.n, tc.m), tensor.WithBacking(matBack))
			rowT := tensor.New(tensor.WithShape(1, tc.m), tensor.WithBacking(rowBack))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g := NewGraph()
				a := NodeFromAny(g, matT, WithName("mat"))
				bN := NodeFromAny(g, rowT, WithName("row"))

				c, err := BroadcastAdd(a, bN, nil, []byte{0})
				if err != nil {
					b.Fatal(err)
				}
				_ = c

				machine := NewTapeMachine(g)
				if err = machine.RunAll(); err != nil {
					b.Fatal(err)
				}
				machine.Close()
			}
		})
	}
}

// BenchmarkBroadcastAdd_ColVec benchmarks adding (N,M) + (N,1) column vector broadcast.
func BenchmarkBroadcastAdd_ColVec(b *testing.B) {
	cases := []struct {
		n, m int
	}{
		{32, 64},
		{128, 256},
		{512, 512},
	}

	for _, tc := range cases {
		name := fmt.Sprintf("N=%d_M=%d", tc.n, tc.m)
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(tc.n * tc.m * 4))

			matBack := makeF32Backing(tc.n * tc.m)
			colBack := makeF32BackingVal(tc.n, 10.0)
			matT := tensor.New(tensor.WithShape(tc.n, tc.m), tensor.WithBacking(matBack))
			colT := tensor.New(tensor.WithShape(tc.n, 1), tensor.WithBacking(colBack))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g := NewGraph()
				a := NodeFromAny(g, matT, WithName("mat"))
				bN := NodeFromAny(g, colT, WithName("col"))

				c, err := BroadcastAdd(a, bN, nil, []byte{1})
				if err != nil {
					b.Fatal(err)
				}
				_ = c

				machine := NewTapeMachine(g)
				if err = machine.RunAll(); err != nil {
					b.Fatal(err)
				}
				machine.Close()
			}
		})
	}
}

// --- Additional Correctness Tests ---

// TestRepeat_LargeMatrix verifies repeat correctness on a 100x100 matrix.
func TestRepeat_LargeMatrix(t *testing.T) {
	assert := assert.New(t)
	const n = 100

	backing := make([]float64, n*n)
	for i := range backing {
		backing[i] = float64(i)
	}
	aT := tensor.New(tensor.WithShape(n, n), tensor.WithBacking(backing))

	g := NewGraph()
	a := NodeFromAny(g, aT, WithName("a"))
	repOp := newRepeatOp(0, a)
	repVal := NewI(3)

	res, err := repOp.Do(aT, repVal)
	if err != nil {
		t.Fatal(err)
	}

	resT := res.(tensor.Tensor)
	assert.Equal(tensor.Shape{300, 100}, resT.Shape())

	data := resT.Data().([]float64)
	for row := 0; row < n; row++ {
		origRow := backing[row*n : (row+1)*n]
		for rep := 0; rep < 3; rep++ {
			resultRow := data[(row*3+rep)*n : (row*3+rep+1)*n]
			assert.Equal(origRow, resultRow, "row=%d rep=%d", row, rep)
		}
	}
}

// TestRepeat_AllAxes systematically tests repeat on each axis of a 3D tensor.
func TestRepeat_AllAxes(t *testing.T) {
	assert := assert.New(t)
	shape := tensor.Shape{2, 3, 4}
	size := shape[0] * shape[1] * shape[2]
	backing := make([]float64, size)
	for i := range backing {
		backing[i] = float64(i)
	}

	for axis := 0; axis < 3; axis++ {
		t.Run(fmt.Sprintf("axis=%d", axis), func(t *testing.T) {
			aT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))

			g := NewGraph()
			a := NodeFromAny(g, aT, WithName("a"))
			repOp := newRepeatOp(axis, a)
			repVal := NewI(2)

			res, err := repOp.Do(aT, repVal)
			if err != nil {
				t.Fatal(err)
			}

			resT := res.(tensor.Tensor)
			expectedShape := shape.Clone()
			expectedShape[axis] *= 2
			assert.Equal(expectedShape, resT.Shape())

			// Verify the total number of elements is correct
			totalExpected := 1
			for _, d := range expectedShape {
				totalExpected *= d
			}
			data := resT.Data().([]float64)
			assert.Equal(totalExpected, len(data))
		})
	}
}

// TestBroadcastAdd_LargeScale verifies numerical correctness of BroadcastAdd at scale.
func TestBroadcastAdd_LargeScale(t *testing.T) {
	assert := assert.New(t)
	const n = 64

	// (N,1) + (1,N) should produce an NxN matrix where result[i][j] = a[i] + b[j]
	aBack := make([]float64, n)
	bBack := make([]float64, n)
	for i := 0; i < n; i++ {
		aBack[i] = float64(i)
		bBack[i] = float64(i * 100)
	}

	aT := tensor.New(tensor.WithShape(n, 1), tensor.WithBacking(aBack))
	bT := tensor.New(tensor.WithShape(1, n), tensor.WithBacking(bBack))

	g := NewGraph()
	a := NodeFromAny(g, aT, WithName("a"))
	b := NodeFromAny(g, bT, WithName("b"))
	c, err := BroadcastAdd(a, b, []byte{1}, []byte{0})
	if err != nil {
		t.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	result := c.Value().Data().([]float64)
	assert.Equal(n*n, len(result))

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := float64(i) + float64(j*100)
			got := result[i*n+j]
			if math.Abs(expected-got) > 1e-10 {
				t.Fatalf("mismatch at [%d,%d]: expected %f, got %f", i, j, expected, got)
			}
		}
	}
}

// TestBroadcastAdd_RowVecColVec verifies row-vector and column-vector broadcast patterns.
func TestBroadcastAdd_RowVecColVec(t *testing.T) {
	assert := assert.New(t)

	t.Run("row_vec_broadcast", func(t *testing.T) {
		// (4,4) + (1,4) => broadcast right on axis 0
		mat := tensor.New(tensor.WithShape(4, 4), tensor.WithBacking([]float64{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}))
		row := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]float64{100, 200, 300, 400}))

		g := NewGraph()
		a := NodeFromAny(g, mat, WithName("mat"))
		b := NodeFromAny(g, row, WithName("row"))

		c, err := BroadcastAdd(a, b, nil, []byte{0})
		if err != nil {
			t.Fatal(err)
		}

		machine := NewTapeMachine(g)
		defer machine.Close()
		if err = machine.RunAll(); err != nil {
			t.Fatal(err)
		}

		expected := []float64{
			101, 202, 303, 404,
			105, 206, 307, 408,
			109, 210, 311, 412,
			113, 214, 315, 416,
		}
		assert.Equal(expected, c.Value().Data().([]float64))
	})

	t.Run("col_vec_broadcast", func(t *testing.T) {
		// (4,4) + (4,1) => broadcast right on axis 1
		mat := tensor.New(tensor.WithShape(4, 4), tensor.WithBacking([]float64{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		}))
		col := tensor.New(tensor.WithShape(4, 1), tensor.WithBacking([]float64{100, 200, 300, 400}))

		g := NewGraph()
		a := NodeFromAny(g, mat, WithName("mat"))
		b := NodeFromAny(g, col, WithName("col"))

		c, err := BroadcastAdd(a, b, nil, []byte{1})
		if err != nil {
			t.Fatal(err)
		}

		machine := NewTapeMachine(g)
		defer machine.Close()
		if err = machine.RunAll(); err != nil {
			t.Fatal(err)
		}

		expected := []float64{
			101, 102, 103, 104,
			205, 206, 207, 208,
			309, 310, 311, 312,
			413, 414, 415, 416,
		}
		assert.Equal(expected, c.Value().Data().([]float64))
	})
}

// TestBroadcastMul_LargeScale verifies numerical correctness of BroadcastHadamardProd at scale.
func TestBroadcastMul_LargeScale(t *testing.T) {
	assert := assert.New(t)
	const n = 64

	// (N,1) * (1,N) should produce an NxN matrix where result[i][j] = a[i] * b[j]
	aBack := make([]float64, n)
	bBack := make([]float64, n)
	for i := 0; i < n; i++ {
		aBack[i] = float64(i + 1)
		bBack[i] = float64(i + 1)
	}

	aT := tensor.New(tensor.WithShape(n, 1), tensor.WithBacking(aBack))
	bT := tensor.New(tensor.WithShape(1, n), tensor.WithBacking(bBack))

	g := NewGraph()
	a := NodeFromAny(g, aT, WithName("a"))
	b := NodeFromAny(g, bT, WithName("b"))
	c, err := BroadcastHadamardProd(a, b, []byte{1}, []byte{0})
	if err != nil {
		t.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	result := c.Value().Data().([]float64)
	assert.Equal(n*n, len(result))

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := float64((i + 1) * (j + 1))
			got := result[i*n+j]
			if math.Abs(expected-got) > 1e-10 {
				t.Fatalf("mismatch at [%d,%d]: expected %f, got %f", i, j, expected, got)
			}
		}
	}
}

// TestRepeat_F32_Correctness verifies repeat with float32 data (the pprof hotspot type).
func TestRepeat_F32_Correctness(t *testing.T) {
	assert := assert.New(t)

	backing := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	aT := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(backing))

	g := NewGraph()
	a := NodeFromAny(g, aT, WithName("a"))
	repOp := newRepeatOp(0, a)
	repVal := NewI(3)

	res, err := repOp.Do(aT, repVal)
	if err != nil {
		t.Fatal(err)
	}

	resT := res.(tensor.Tensor)
	assert.Equal(tensor.Shape{6, 3}, resT.Shape())

	expected := []float32{
		1, 2, 3, // row 0, rep 0
		1, 2, 3, // row 0, rep 1
		1, 2, 3, // row 0, rep 2
		4, 5, 6, // row 1, rep 0
		4, 5, 6, // row 1, rep 1
		4, 5, 6, // row 1, rep 2
	}
	assert.Equal(expected, resT.Data().([]float32))
}
