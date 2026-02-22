package gorgonia

import (
	"fmt"
	"math"
	"testing"

	"gorgonia.org/tensor"
)

// benchmarkMatMul runs a matrix multiply benchmark for the given dtype and dimensions.
// It builds a graph once and reuses it via TapeMachine.Reset() across iterations.
func benchmarkMatMul(b *testing.B, dt tensor.Dtype, m, k, n int) {
	b.Helper()
	g := NewGraph()

	var elemSize int
	switch dt {
	case Float32:
		elemSize = 4
	case Float64:
		elemSize = 8
	}

	aBack := tensor.Random(dt, m*k)
	bBack := tensor.Random(dt, k*n)
	aNode := NewMatrix(g, dt, WithShape(m, k), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(m, k), tensor.WithBacking(aBack)),
	))
	bNode := NewMatrix(g, dt, WithShape(k, n), WithName("b"), WithValue(
		tensor.New(tensor.WithShape(k, n), tensor.WithBacking(bBack)),
	))
	_, err := Mul(aNode, bNode)
	if err != nil {
		b.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		b.Fatal(err)
	}

	// 2*m*k*n flops for matmul
	flops := int64(2) * int64(m) * int64(k) * int64(n)
	b.SetBytes(int64(elemSize) * (int64(m*k) + int64(k*n) + int64(m*n)))
	b.ReportAllocs()
	b.ReportMetric(float64(flops), "flops/op")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		machine.Reset()
		if err := machine.RunAll(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMul_F32(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			benchmarkMatMul(b, Float32, n, n, n)
		})
	}
}

func BenchmarkMatMul_F64(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			benchmarkMatMul(b, Float64, n, n, n)
		})
	}
}

func BenchmarkMatMul_F32_NonSquare(b *testing.B) {
	cases := []struct {
		m, k, n int
	}{
		{128, 64, 256},
		{512, 32, 512},
		{64, 512, 64},
		{1024, 64, 128},
	}
	for _, c := range cases {
		b.Run(fmt.Sprintf("%dx%d_x_%dx%d", c.m, c.k, c.k, c.n), func(b *testing.B) {
			benchmarkMatMul(b, Float32, c.m, c.k, c.n)
		})
	}
}

// benchmarkMatVecMul runs a matrix-vector multiplication benchmark.
func benchmarkMatVecMul(b *testing.B, dt tensor.Dtype, n int) {
	b.Helper()
	g := NewGraph()

	var elemSize int
	switch dt {
	case Float32:
		elemSize = 4
	case Float64:
		elemSize = 8
	}

	matBack := tensor.Random(dt, n*n)
	vecBack := tensor.Random(dt, n)
	matNode := NewMatrix(g, dt, WithShape(n, n), WithName("mat"), WithValue(
		tensor.New(tensor.WithShape(n, n), tensor.WithBacking(matBack)),
	))
	vecNode := NewVector(g, dt, WithShape(n), WithName("vec"), WithValue(
		tensor.New(tensor.WithShape(n), tensor.WithBacking(vecBack)),
	))
	_, err := Mul(matNode, vecNode)
	if err != nil {
		b.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		b.Fatal(err)
	}

	flops := int64(2) * int64(n) * int64(n)
	b.SetBytes(int64(elemSize) * (int64(n*n) + int64(n) + int64(n)))
	b.ReportAllocs()
	b.ReportMetric(float64(flops), "flops/op")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		machine.Reset()
		if err := machine.RunAll(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecMul_F32(b *testing.B) {
	sizes := []int{64, 256, 1024}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			benchmarkMatVecMul(b, Float32, n)
		})
	}
}

func BenchmarkMatVecMul_F64(b *testing.B) {
	sizes := []int{64, 256, 1024}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			benchmarkMatVecMul(b, Float64, n)
		})
	}
}

// benchmarkVecDot runs a vector dot product benchmark.
func benchmarkVecDot(b *testing.B, dt tensor.Dtype, n int) {
	b.Helper()
	g := NewGraph()

	var elemSize int
	switch dt {
	case Float32:
		elemSize = 4
	case Float64:
		elemSize = 8
	}

	aBack := tensor.Random(dt, n)
	bBack := tensor.Random(dt, n)
	aNode := NewVector(g, dt, WithShape(n), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(n), tensor.WithBacking(aBack)),
	))
	bNode := NewVector(g, dt, WithShape(n), WithName("b"), WithValue(
		tensor.New(tensor.WithShape(n), tensor.WithBacking(bBack)),
	))
	_, err := Mul(aNode, bNode)
	if err != nil {
		b.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		b.Fatal(err)
	}

	flops := int64(2) * int64(n)
	b.SetBytes(int64(elemSize) * int64(2*n))
	b.ReportAllocs()
	b.ReportMetric(float64(flops), "flops/op")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		machine.Reset()
		if err := machine.RunAll(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVecDot_F32(b *testing.B) {
	sizes := []int{1000, 10000, 100000}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			benchmarkVecDot(b, Float32, n)
		})
	}
}

// benchmarkOuterProd runs an outer product benchmark.
func benchmarkOuterProd(b *testing.B, dt tensor.Dtype, n int) {
	b.Helper()
	g := NewGraph()

	var elemSize int
	switch dt {
	case Float32:
		elemSize = 4
	case Float64:
		elemSize = 8
	}

	aBack := tensor.Random(dt, n)
	bBack := tensor.Random(dt, n)
	aNode := NewVector(g, dt, WithShape(n), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(n), tensor.WithBacking(aBack)),
	))
	bNode := NewVector(g, dt, WithShape(n), WithName("b"), WithValue(
		tensor.New(tensor.WithShape(n), tensor.WithBacking(bBack)),
	))
	_, err := OuterProd(aNode, bNode)
	if err != nil {
		b.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		b.Fatal(err)
	}

	flops := int64(n) * int64(n)
	b.SetBytes(int64(elemSize) * (int64(2*n) + int64(n*n)))
	b.ReportAllocs()
	b.ReportMetric(float64(flops), "flops/op")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		machine.Reset()
		if err := machine.RunAll(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOuterProd_F32(b *testing.B) {
	sizes := []int{64, 256, 1024}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			benchmarkOuterProd(b, Float32, n)
		})
	}
}

// benchmarkBatchedMatMul runs a batched matrix multiply benchmark.
func benchmarkBatchedMatMul(b *testing.B, dt tensor.Dtype, batch, m, k, n int) {
	b.Helper()
	g := NewGraph()

	var elemSize int
	switch dt {
	case Float32:
		elemSize = 4
	case Float64:
		elemSize = 8
	}

	aBack := tensor.Random(dt, batch*m*k)
	bBack := tensor.Random(dt, batch*k*n)
	aNode := NewTensor(g, dt, 3, WithShape(batch, m, k), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(batch, m, k), tensor.WithBacking(aBack)),
	))
	bNode := NewTensor(g, dt, 3, WithShape(batch, k, n), WithName("b"), WithValue(
		tensor.New(tensor.WithShape(batch, k, n), tensor.WithBacking(bBack)),
	))
	_, err := BatchedMatMul(aNode, bNode)
	if err != nil {
		b.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		b.Fatal(err)
	}

	flops := int64(2) * int64(batch) * int64(m) * int64(k) * int64(n)
	b.SetBytes(int64(elemSize) * int64(batch) * (int64(m*k) + int64(k*n) + int64(m*n)))
	b.ReportAllocs()
	b.ReportMetric(float64(flops), "flops/op")
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		machine.Reset()
		if err := machine.RunAll(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBatchedMatMul_F32(b *testing.B) {
	batches := []int{1, 4, 16}
	for _, batch := range batches {
		b.Run(fmt.Sprintf("batch=%d_64x64", batch), func(b *testing.B) {
			benchmarkBatchedMatMul(b, Float32, batch, 64, 64, 64)
		})
	}
}

// BenchmarkConv2d_Sgemm benchmarks Conv2d which exercises SGEMM internally.
// Scaled up from the TestUse pattern in blas_test.go.
func BenchmarkConv2d_Sgemm(b *testing.B) {
	cases := []struct {
		name              string
		batch, ch, h, w   int
		fOut, fIn, fH, fW int
		kernel            tensor.Shape
		pad, stride, dil  []int
	}{
		{"small_1x1x7x5", 1, 1, 7, 5, 1, 1, 3, 3, tensor.Shape{3, 3}, []int{0, 0}, []int{2, 2}, []int{1, 1}},
		{"medium_1x3x28x28", 1, 3, 28, 28, 16, 3, 3, 3, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}},
		{"large_4x16x32x32", 4, 16, 32, 32, 32, 16, 3, 3, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			g := NewGraph()
			xBack := tensor.Random(Float32, c.batch*c.ch*c.h*c.w)
			fBack := tensor.Random(Float32, c.fOut*c.fIn*c.fH*c.fW)
			x := NodeFromAny(g, tensor.New(
				tensor.WithShape(c.batch, c.ch, c.h, c.w),
				tensor.WithBacking(xBack),
			))
			filter := NodeFromAny(g, tensor.New(
				tensor.WithShape(c.fOut, c.fIn, c.fH, c.fW),
				tensor.WithBacking(fBack),
			))
			_, err := Conv2d(x, filter, c.kernel, c.pad, c.stride, c.dil)
			if err != nil {
				b.Fatal(err)
			}

			machine := NewTapeMachine(g)
			defer machine.Close()
			if err := machine.RunAll(); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				machine.Reset()
				if err := machine.RunAll(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// --- Correctness Tests ---

// TestMatMul_F32_Correctness verifies matmul against hand-computed results.
func TestMatMul_F32_Correctness(t *testing.T) {
	cases := []struct {
		name     string
		m, k, n  int
		aData    []float32
		bData    []float32
		expected []float32
	}{
		{
			name: "2x3_x_3x2",
			m:    2, k: 3, n: 2,
			aData:    []float32{1, 2, 3, 4, 5, 6},
			bData:    []float32{7, 8, 9, 10, 11, 12},
			expected: []float32{58, 64, 139, 154},
		},
		{
			name: "1x1_x_1x1",
			m:    1, k: 1, n: 1,
			aData:    []float32{3},
			bData:    []float32{7},
			expected: []float32{21},
		},
		{
			name: "3x3_x_3x3",
			m:    3, k: 3, n: 3,
			aData:    []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}, // identity
			bData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGraph()
			a := NewMatrix(g, Float32, WithShape(tc.m, tc.k), WithName("a"), WithValue(
				tensor.New(tensor.WithShape(tc.m, tc.k), tensor.WithBacking(tc.aData)),
			))
			b := NewMatrix(g, Float32, WithShape(tc.k, tc.n), WithName("b"), WithValue(
				tensor.New(tensor.WithShape(tc.k, tc.n), tensor.WithBacking(tc.bData)),
			))
			c, err := Mul(a, b)
			if err != nil {
				t.Fatal(err)
			}
			machine := NewTapeMachine(g)
			defer machine.Close()
			if err := machine.RunAll(); err != nil {
				t.Fatal(err)
			}

			result := c.Value().Data().([]float32)
			if len(result) != len(tc.expected) {
				t.Fatalf("result length %d, want %d", len(result), len(tc.expected))
			}
			for i := range result {
				if result[i] != tc.expected[i] {
					t.Errorf("result[%d] = %f, want %f", i, result[i], tc.expected[i])
				}
			}
		})
	}
}

// TestMatMul_F64_Correctness verifies F64 matmul at a larger size against naive computation.
func TestMatMul_F64_Correctness(t *testing.T) {
	m, k, n := 16, 32, 16
	aData := make([]float64, m*k)
	bData := make([]float64, k*n)
	for i := range aData {
		aData[i] = float64(i + 1)
	}
	for i := range bData {
		bData[i] = float64(i + 1)
	}

	// Compute expected result naively
	expected := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for l := 0; l < k; l++ {
				sum += aData[i*k+l] * bData[l*n+j]
			}
			expected[i*n+j] = sum
		}
	}

	g := NewGraph()
	a := NewMatrix(g, Float64, WithShape(m, k), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(m, k), tensor.WithBacking(aData)),
	))
	b := NewMatrix(g, Float64, WithShape(k, n), WithName("b"), WithValue(
		tensor.New(tensor.WithShape(k, n), tensor.WithBacking(bData)),
	))
	c, err := Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	result := c.Value().Data().([]float64)
	for i := range result {
		if math.Abs(result[i]-expected[i]) > 1e-6 {
			t.Errorf("result[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

// TestMatMul_F32_Identity verifies that multiplying by the identity matrix returns the original.
func TestMatMul_F32_Identity(t *testing.T) {
	sizes := []int{2, 4, 8, 16}
	for _, n := range sizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			aData := make([]float32, n*n)
			for i := range aData {
				aData[i] = float32(i + 1)
			}
			identity := make([]float32, n*n)
			for i := 0; i < n; i++ {
				identity[i*n+i] = 1
			}

			g := NewGraph()
			a := NewMatrix(g, Float32, WithShape(n, n), WithName("a"), WithValue(
				tensor.New(tensor.WithShape(n, n), tensor.WithBacking(aData)),
			))
			eye := NewMatrix(g, Float32, WithShape(n, n), WithName("eye"), WithValue(
				tensor.New(tensor.WithShape(n, n), tensor.WithBacking(identity)),
			))
			c, err := Mul(a, eye)
			if err != nil {
				t.Fatal(err)
			}
			machine := NewTapeMachine(g)
			defer machine.Close()
			if err := machine.RunAll(); err != nil {
				t.Fatal(err)
			}

			result := c.Value().Data().([]float32)
			for i := range result {
				if result[i] != aData[i] {
					t.Errorf("result[%d] = %f, want %f", i, result[i], aData[i])
				}
			}
		})
	}
}

// TestMatMul_F32_Transpose verifies matmul with transpose combinations using BatchedMatMul.
func TestMatMul_F32_Transpose(t *testing.T) {
	// A = [[1,2],[3,4]] shape (1,2,2)
	// B = [[5,6],[7,8]] shape (1,2,2)
	aData := []float64{1, 2, 3, 4}
	bData := []float64{5, 6, 7, 8}

	cases := []struct {
		name     string
		transA   bool
		transB   bool
		expected []float64
	}{
		{
			name:     "NoTrans_NoTrans",
			transA:   false,
			transB:   false,
			expected: []float64{19, 22, 43, 50}, // A*B
		},
		{
			name:     "TransA_NoTrans",
			transA:   true,
			transB:   false,
			expected: []float64{26, 30, 38, 44}, // A^T * B
		},
		{
			name:     "NoTrans_TransB",
			transA:   false,
			transB:   true,
			expected: []float64{17, 23, 39, 53}, // A * B^T
		},
		{
			name:     "TransA_TransB",
			transA:   true,
			transB:   true,
			expected: []float64{23, 31, 34, 46}, // A^T * B^T
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGraph()
			a := NewTensor(g, Float64, 3, WithShape(1, 2, 2), WithName("a"), WithValue(
				tensor.New(tensor.WithShape(1, 2, 2), tensor.WithBacking(append([]float64{}, aData...))),
			))
			b := NewTensor(g, Float64, 3, WithShape(1, 2, 2), WithName("b"), WithValue(
				tensor.New(tensor.WithShape(1, 2, 2), tensor.WithBacking(append([]float64{}, bData...))),
			))
			c, err := BatchedMatMul(a, b, tc.transA, tc.transB)
			if err != nil {
				t.Fatal(err)
			}
			machine := NewTapeMachine(g)
			defer machine.Close()
			if err := machine.RunAll(); err != nil {
				t.Fatal(err)
			}

			result := c.Value().Data().([]float64)
			if len(result) != len(tc.expected) {
				t.Fatalf("result length %d, want %d", len(result), len(tc.expected))
			}
			for i := range result {
				if math.Abs(result[i]-tc.expected[i]) > 1e-6 {
					t.Errorf("result[%d] = %f, want %f", i, result[i], tc.expected[i])
				}
			}
		})
	}
}

// TestBatchedMatMul_Shapes tests various batch dimension configurations.
func TestBatchedMatMul_Shapes(t *testing.T) {
	cases := []struct {
		name           string
		dims           int
		aShape, bShape tensor.Shape
	}{
		{"batch=2_2x3_3x2", 3, tensor.Shape{2, 2, 3}, tensor.Shape{2, 3, 2}},
		{"batch=5_1x4_4x1", 3, tensor.Shape{5, 1, 4}, tensor.Shape{5, 4, 1}},
		{"batch=1x3_2x4_4x2", 4, tensor.Shape{1, 3, 2, 4}, tensor.Shape{1, 3, 4, 2}},
		{"batch=3x2_2x3_3x2", 4, tensor.Shape{3, 2, 2, 3}, tensor.Shape{3, 2, 3, 2}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGraph()
			a := NewTensor(g, Float64, tc.dims, WithShape(tc.aShape...), WithName("a"), WithInit(RangedFrom(1)))
			b := NewTensor(g, Float64, tc.dims, WithShape(tc.bShape...), WithName("b"), WithInit(RangedFrom(1)))
			c, err := BatchedMatMul(a, b)
			if err != nil {
				t.Fatal(err)
			}
			machine := NewTapeMachine(g)
			defer machine.Close()
			if err := machine.RunAll(); err != nil {
				t.Fatal(err)
			}

			cShape := c.Value().Shape()
			// Verify batch dimensions match
			for i := 0; i < tc.dims-2; i++ {
				if cShape[i] != tc.aShape[i] {
					t.Errorf("batch dim %d: got %d, want %d", i, cShape[i], tc.aShape[i])
				}
			}
			// Verify result matrix dimensions: rows from A, cols from B
			if cShape[tc.dims-2] != tc.aShape[tc.dims-2] {
				t.Errorf("result rows: got %d, want %d", cShape[tc.dims-2], tc.aShape[tc.dims-2])
			}
			if cShape[tc.dims-1] != tc.bShape[tc.dims-1] {
				t.Errorf("result cols: got %d, want %d", cShape[tc.dims-1], tc.bShape[tc.dims-1])
			}
		})
	}
}

// TestMatMul_Gradient verifies that the backward pass produces correct gradients.
func TestMatMul_Gradient(t *testing.T) {
	// For C = A * B, dL/dA = dL/dC * B^T, dL/dB = A^T * dL/dC
	// With Sum as loss, dL/dC = ones matrix
	g := NewGraph()
	aData := []float64{1, 2, 3, 4, 5, 6}
	bData := []float64{7, 8, 9, 10, 11, 12}

	a := NewMatrix(g, Float64, WithShape(2, 3), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(aData)),
	))
	b := NewMatrix(g, Float64, WithShape(3, 2), WithName("b"), WithValue(
		tensor.New(tensor.WithShape(3, 2), tensor.WithBacking(bData)),
	))
	c, err := Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := Sum(c)
	if err != nil {
		t.Fatal(err)
	}
	grads, err := Grad(loss, a, b)
	if err != nil {
		t.Fatal(err)
	}

	machine := NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	// dL/dA = ones(2,2) * B^T
	// B^T = [[7,9,11],[8,10,12]]
	// ones(2,2) * B^T = [[15,19,23],[15,19,23]]
	expectedGradA := []float64{15, 19, 23, 15, 19, 23}
	gradA := grads[0].Value().Data().([]float64)
	for i := range gradA {
		if math.Abs(gradA[i]-expectedGradA[i]) > 1e-6 {
			t.Errorf("gradA[%d] = %f, want %f", i, gradA[i], expectedGradA[i])
		}
	}

	// dL/dB = A^T * ones(2,2)
	// A^T = [[1,4],[2,5],[3,6]]
	// A^T * ones(2,2) = [[5,5],[7,7],[9,9]]
	expectedGradB := []float64{5, 5, 7, 7, 9, 9}
	gradB := grads[1].Value().Data().([]float64)
	for i := range gradB {
		if math.Abs(gradB[i]-expectedGradB[i]) > 1e-6 {
			t.Errorf("gradB[%d] = %f, want %f", i, gradB[i], expectedGradB[i])
		}
	}
}
