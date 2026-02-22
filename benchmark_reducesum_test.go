package gorgonia

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

// --- Benchmarks ---

// BenchmarkSum_F32_Vector benchmarks full-reduce Sum on float32 vectors of increasing size.
func BenchmarkSum_F32_Vector(b *testing.B) {
	sizes := []int{1_000, 10_000, 100_000, 1_000_000}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			backing := make([]float32, n)
			for i := range backing {
				backing[i] = float32(i)
			}
			g := NewGraph()
			x := NewTensor(g, Float32, 1, WithShape(n))
			Must(Sum(x, 0))

			xT := tensor.New(tensor.WithShape(n), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(n * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// BenchmarkSum_F64_Vector benchmarks full-reduce Sum on float64 vectors of increasing size.
func BenchmarkSum_F64_Vector(b *testing.B) {
	sizes := []int{1_000, 10_000, 100_000, 1_000_000}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			backing := make([]float64, n)
			for i := range backing {
				backing[i] = float64(i)
			}
			g := NewGraph()
			x := NewTensor(g, Float64, 1, WithShape(n))
			Must(Sum(x, 0))

			xT := tensor.New(tensor.WithShape(n), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// BenchmarkSum_F32_Matrix_Axis0 benchmarks Sum along axis 0 for square float32 matrices.
func BenchmarkSum_F32_Matrix_Axis0(b *testing.B) {
	dims := []int{32, 128, 512, 1024}
	for _, n := range dims {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			size := n * n
			backing := make([]float32, size)
			for i := range backing {
				backing[i] = float32(i)
			}
			g := NewGraph()
			x := NewTensor(g, Float32, 2, WithShape(n, n))
			Must(Sum(x, 0))

			xT := tensor.New(tensor.WithShape(n, n), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// BenchmarkSum_F32_Matrix_Axis1 benchmarks Sum along axis 1 for square float32 matrices.
func BenchmarkSum_F32_Matrix_Axis1(b *testing.B) {
	dims := []int{32, 128, 512, 1024}
	for _, n := range dims {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			size := n * n
			backing := make([]float32, size)
			for i := range backing {
				backing[i] = float32(i)
			}
			g := NewGraph()
			x := NewTensor(g, Float32, 2, WithShape(n, n))
			Must(Sum(x, 1))

			xT := tensor.New(tensor.WithShape(n, n), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// BenchmarkSum_F32_FullReduce benchmarks full reduction (all axes) for float32 matrices.
func BenchmarkSum_F32_FullReduce(b *testing.B) {
	dims := []int{32, 128, 512, 1024}
	for _, n := range dims {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			size := n * n
			backing := make([]float32, size)
			for i := range backing {
				backing[i] = float32(i)
			}
			g := NewGraph()
			x := NewTensor(g, Float32, 2, WithShape(n, n))
			Must(Sum(x))

			xT := tensor.New(tensor.WithShape(n, n), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// BenchmarkSum_F32_4D benchmarks Sum along each single axis of a 4D float32 tensor.
func BenchmarkSum_F32_4D(b *testing.B) {
	shape := []int{8, 16, 32, 64}
	size := 8 * 16 * 32 * 64
	backing := make([]float32, size)
	for i := range backing {
		backing[i] = float32(i)
	}

	for axis := 0; axis < 4; axis++ {
		b.Run(fmt.Sprintf("axis=%d", axis), func(b *testing.B) {
			g := NewGraph()
			x := NewTensor(g, Float32, 4, WithShape(shape...))
			Must(Sum(x, axis))

			xT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// BenchmarkSum_F32_MultiAxis benchmarks Sum along multiple axes of a 4D float32 tensor.
func BenchmarkSum_F32_MultiAxis(b *testing.B) {
	shape := []int{8, 16, 32, 64}
	size := 8 * 16 * 32 * 64
	backing := make([]float32, size)
	for i := range backing {
		backing[i] = float32(i)
	}

	cases := []struct {
		name string
		axes []int
	}{
		{"axes_0_2", []int{0, 2}},
		{"axes_1_3", []int{1, 3}},
		{"axes_0_1_2", []int{0, 1, 2}},
		{"axes_all", []int{0, 1, 2, 3}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			g := NewGraph()
			x := NewTensor(g, Float32, 4, WithShape(shape...))
			Must(Sum(x, tc.axes...))

			xT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))
			vm := NewTapeMachine(g)
			defer vm.Close()
			vm.Let(x, xT)

			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := vm.RunAll(); err != nil {
					b.Fatal(err)
				}
				vm.Reset()
			}
		})
	}
}

// --- Additional Correctness Tests ---

// TestSum_F64 mirrors the existing Float32 TestSumOp tests for Float64.
func TestSum_F64(t *testing.T) {
	subTests := []reductionTest{
		{dt: Float64, inShape: []int{3, 2}, inData: []float64{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{0}, wantShape: []int{2}, wantData: []float64{9, 12}},
		{dt: Float64, inShape: []int{3, 2}, inData: []float64{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{1}, wantShape: []int{3}, wantData: []float64{3, 7, 11}},
		{dt: Float64, inShape: []int{3, 2}, inData: []float64{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{}, wantShape: []int{}, wantData: float64(21)},
		{dt: Float64, inShape: []int{3, 2}, inData: []float64{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{0, 1}, wantShape: []int{}, wantData: float64(21)},
		{dt: Float64, inShape: []int{3, 2}, inData: []float64{1, 2, 3, 4, 5, 6}, op: Sum, along: []int{1, 0}, wantShape: []int{}, wantData: float64(21)},
		{
			dt:        Float64,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{0},
			wantShape: []int{2, 2, 2},
			wantData:  []float64{10, 12, 14, 16, 18, 20, 22, 24},
		},
		{
			dt:        Float64,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{1, 3},
			wantShape: []int{2, 2},
			wantData:  []float64{14, 22, 46, 54},
		},
		{
			dt:        Float64,
			inShape:   []int{2, 2, 2, 2},
			inData:    []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			op:        Sum,
			along:     []int{0, 2, 3},
			wantShape: []int{2},
			wantData:  []float64{52, 84},
		},
	}

	for _, subTest := range subTests {
		t.Run(fmt.Sprintf("along_%v", subTest.along), func(t *testing.T) {
			testReductionOp(t, subTest)
		})
	}
}

// TestSum_LargeMatrix verifies correctness of Sum on a 100x100 matrix.
func TestSum_LargeMatrix(t *testing.T) {
	const rows, cols = 100, 100
	size := rows * cols
	backing := make([]float64, size)
	for i := range backing {
		backing[i] = float64(i)
	}

	// Full reduction: sum of 0..9999 = 9999*10000/2 = 49995000
	t.Run("full_reduce", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 2, WithShape(rows, cols))
		got := Must(Sum(x))

		xT := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, float64(49995000), got.Value().Data())
	})

	// Sum along axis 0: each column j sums rows 0..99 -> sum = sum_{r=0}^{99}(r*100 + j)
	t.Run("axis_0", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 2, WithShape(rows, cols))
		got := Must(Sum(x, 0))

		xT := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}

		result := got.Value().Data().([]float64)
		assert.Equal(t, cols, len(result))
		// Column 0: 0+100+200+...+9900 = 100*(0+1+...+99) = 100*4950 = 495000
		assert.Equal(t, float64(495000), result[0])
		// Column 99: 99+199+...+9999 = 495000 + 99*100 = 504900
		assert.Equal(t, float64(504900), result[cols-1])
	})

	// Sum along axis 1: each row r sums cols 0..99 -> sum = sum_{c=0}^{99}(r*100 + c)
	t.Run("axis_1", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 2, WithShape(rows, cols))
		got := Must(Sum(x, 1))

		xT := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}

		result := got.Value().Data().([]float64)
		assert.Equal(t, rows, len(result))
		// Row 0: 0+1+...+99 = 4950
		assert.Equal(t, float64(4950), result[0])
		// Row 99: 9900+9901+...+9999 = 99*100*100 + 4950 = 994950
		assert.Equal(t, float64(994950), result[rows-1])
	})
}

// TestSum_HighDimensional verifies correctness on a 5D tensor reduction.
func TestSum_HighDimensional(t *testing.T) {
	shape := []int{2, 3, 4, 5, 6}
	size := 2 * 3 * 4 * 5 * 6 // 720
	backing := make([]float64, size)
	for i := range backing {
		backing[i] = float64(i)
	}

	// Full reduction: verify shape is scalar and value matches naive sum.
	t.Run("full_reduce", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 5, WithShape(shape...))
		got := Must(Sum(x))

		xT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		assert.True(t, got.Value().Shape().IsScalar(), "full reduce should yield scalar")
		// Verify against naive sum to account for multi-axis reduction ordering.
		var naiveSum float64
		for _, v := range backing {
			naiveSum += v
		}
		assert.InDelta(t, naiveSum, got.Value().Data().(float64), 1e-6)
	})

	// Reduce along axis 0 (size 2): result shape (3,4,5,6)
	t.Run("axis_0", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 5, WithShape(shape...))
		got := Must(Sum(x, 0))

		xT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, tensor.Shape{3, 4, 5, 6}, got.Value().Shape())
		result := got.Value().Data().([]float64)
		assert.Equal(t, 3*4*5*6, len(result))
		// First element: backing[0] + backing[360] = 0 + 360 = 360
		assert.Equal(t, float64(360), result[0])
	})

	// Reduce along axis 4 (size 6): result shape (2,3,4,5)
	t.Run("axis_4", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 5, WithShape(shape...))
		got := Must(Sum(x, 4))

		xT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, tensor.Shape{2, 3, 4, 5}, got.Value().Shape())
		result := got.Value().Data().([]float64)
		assert.Equal(t, 2*3*4*5, len(result))
		// First element: sum of backing[0:6] = 0+1+2+3+4+5 = 15
		assert.Equal(t, float64(15), result[0])
	})

	// Reduce along multiple axes (0, 2)
	t.Run("axes_0_2", func(t *testing.T) {
		g := NewGraph()
		x := NewTensor(g, Float64, 5, WithShape(shape...))
		got := Must(Sum(x, 0, 2))

		xT := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, tensor.Shape{3, 5, 6}, got.Value().Shape())
	})
}

// TestSum_SingleElementAxis verifies reduction along an axis of size 1.
func TestSum_SingleElementAxis(t *testing.T) {
	subTests := []reductionTest{
		// Shape (1, 4): reduce along axis 0 (size 1) -> shape (4)
		{
			dt:        Float32,
			inShape:   []int{1, 4},
			inData:    []float32{10, 20, 30, 40},
			op:        Sum,
			along:     []int{0},
			wantShape: []int{4},
			wantData:  []float32{10, 20, 30, 40},
		},
		// Shape (4, 1): reduce along axis 1 (size 1) -> shape (4)
		{
			dt:        Float32,
			inShape:   []int{4, 1},
			inData:    []float32{10, 20, 30, 40},
			op:        Sum,
			along:     []int{1},
			wantShape: []int{4},
			wantData:  []float32{10, 20, 30, 40},
		},
		// Shape (1, 1, 4): reduce along axis 0 (size 1) -> shape (1, 4)
		{
			dt:        Float64,
			inShape:   []int{1, 1, 4},
			inData:    []float64{1, 2, 3, 4},
			op:        Sum,
			along:     []int{0},
			wantShape: []int{1, 4},
			wantData:  []float64{1, 2, 3, 4},
		},
		// Shape (2, 1, 3): reduce along axis 1 (size 1) -> shape (2, 3)
		{
			dt:        Float64,
			inShape:   []int{2, 1, 3},
			inData:    []float64{1, 2, 3, 4, 5, 6},
			op:        Sum,
			along:     []int{1},
			wantShape: []int{2, 3},
			wantData:  []float64{1, 2, 3, 4, 5, 6},
		},
	}

	for _, subTest := range subTests {
		t.Run(fmt.Sprintf("shape_%v_along_%v", subTest.inShape, subTest.along), func(t *testing.T) {
			testReductionOp(t, subTest)
		})
	}
}

// TestSum_F64_4D tests Float64 sum on a 4D tensor with multi-axis reduction to confirm
// parity with the Float32 tests in TestSumOp.
func TestSum_F64_4D(t *testing.T) {
	data := make([]float64, 16)
	for i := range data {
		data[i] = float64(i + 1)
	}

	subTests := []reductionTest{
		{
			dt: Float64, inShape: []int{2, 2, 2, 2}, inData: data, op: Sum,
			along: []int{0, 1, 2, 3}, wantShape: []int{}, wantData: float64(136),
		},
		{
			dt: Float64, inShape: []int{2, 2, 2, 2}, inData: data, op: Sum,
			along: []int{}, wantShape: []int{}, wantData: float64(136),
		},
		{
			dt: Float64, inShape: []int{2, 2, 2, 2}, inData: data, op: Sum,
			along: []int{3}, wantShape: []int{2, 2, 2},
			wantData: []float64{3, 7, 11, 15, 19, 23, 27, 31},
		},
		{
			dt: Float64, inShape: []int{2, 2, 2, 2}, inData: data, op: Sum,
			along: []int{1, 3}, wantShape: []int{2, 2},
			wantData: []float64{14, 22, 46, 54},
		},
	}

	for _, subTest := range subTests {
		t.Run(fmt.Sprintf("along_%v", subTest.along), func(t *testing.T) {
			testReductionOp(t, subTest)
		})
	}
}

// TestSum_LargeVector_Accuracy checks that summing a large float32 vector does not lose
// excessive precision (a baseline for any future SIMD or tiling optimizations).
func TestSum_LargeVector_Accuracy(t *testing.T) {
	const n = 100_000
	backing := make([]float32, n)
	for i := range backing {
		backing[i] = 1.0
	}

	g := NewGraph()
	x := NewTensor(g, Float32, 1, WithShape(n))
	got := Must(Sum(x, 0))

	xT := tensor.New(tensor.WithShape(n), tensor.WithBacking(backing))
	vm := NewTapeMachine(g)
	defer vm.Close()
	vm.Let(x, xT)
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}

	result := got.Value().Data().(float32)
	// Allow some float32 accumulation error, but expect within 0.1% of exact.
	assert.InDelta(t, float64(n), float64(result), float64(n)*0.001,
		"sum of %d ones should be close to %d, got %f", n, n, result)
}

// TestSum_EmptyAlong_EquivFullReduce confirms that Sum with no along args
// is equivalent to Sum with all axes listed.
func TestSum_EmptyAlong_EquivFullReduce(t *testing.T) {
	backing := make([]float64, 24)
	for i := range backing {
		backing[i] = float64(i + 1)
	}

	// Sum with no axes
	g1 := NewGraph()
	x1 := NewTensor(g1, Float64, 3, WithShape(2, 3, 4))
	got1 := Must(Sum(x1))
	xT1 := tensor.New(tensor.WithShape(2, 3, 4), tensor.WithBacking(backing))
	vm1 := NewTapeMachine(g1)
	defer vm1.Close()
	vm1.Let(x1, xT1)
	if err := vm1.RunAll(); err != nil {
		t.Fatal(err)
	}

	// Sum with all axes
	g2 := NewGraph()
	x2 := NewTensor(g2, Float64, 3, WithShape(2, 3, 4))
	got2 := Must(Sum(x2, 0, 1, 2))
	xT2 := tensor.New(tensor.WithShape(2, 3, 4), tensor.WithBacking(backing))
	vm2 := NewTapeMachine(g2)
	defer vm2.Close()
	vm2.Let(x2, xT2)
	if err := vm2.RunAll(); err != nil {
		t.Fatal(err)
	}

	v1 := got1.Value().Data().(float64)
	v2 := got2.Value().Data().(float64)
	assert.Equal(t, v1, v2, "Sum() and Sum(0,1,2) should produce the same result")
	// sum of 1..24 = 300
	assert.Equal(t, float64(300), v1)
}

// TestSum_F32_Deterministic verifies that running the same Sum twice yields identical results.
func TestSum_F32_Deterministic(t *testing.T) {
	const n = 10_000
	backing := make([]float32, n)
	for i := range backing {
		backing[i] = float32(math.Sin(float64(i)))
	}

	run := func() float32 {
		g := NewGraph()
		x := NewTensor(g, Float32, 1, WithShape(n))
		got := Must(Sum(x, 0))
		xT := tensor.New(tensor.WithShape(n), tensor.WithBacking(backing))
		vm := NewTapeMachine(g)
		defer vm.Close()
		vm.Let(x, xT)
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		return got.Value().Data().(float32)
	}

	r1 := run()
	r2 := run()
	assert.Equal(t, r1, r2, "repeated Sum should be deterministic")
}
