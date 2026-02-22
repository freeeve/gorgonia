package gorgonia

import (
	"fmt"
	"math"
	"testing"

	"gorgonia.org/tensor"
)

func TestNEON_MatMul(t *testing.T) {
	impl := NEONImplementation()
	if impl == nil {
		t.Skip("NEON not available on this platform")
	}

	oldBLAS := WhichBLAS()
	Use(impl)
	defer Use(oldBLAS)

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
			name: "identity_3x3",
			m:    3, k: 3, n: 3,
			aData:    []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
			bData:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			expected: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGraph()
			a := NewMatrix(g, Float32, WithShape(tc.m, tc.k), WithName("a"), WithValue(
				tensor.New(tensor.WithShape(tc.m, tc.k), tensor.WithBacking(append([]float32{}, tc.aData...))),
			))
			b := NewMatrix(g, Float32, WithShape(tc.k, tc.n), WithName("b"), WithValue(
				tensor.New(tensor.WithShape(tc.k, tc.n), tensor.WithBacking(append([]float32{}, tc.bData...))),
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
				if math.Abs(float64(result[i]-tc.expected[i])) > 1e-3 {
					t.Errorf("result[%d] = %f, want %f", i, result[i], tc.expected[i])
				}
			}
		})
	}
}

func TestNEON_LargerMatMul(t *testing.T) {
	impl := NEONImplementation()
	if impl == nil {
		t.Skip("NEON not available on this platform")
	}

	oldBLAS := WhichBLAS()
	Use(impl)
	defer Use(oldBLAS)

	m, k, n := 32, 64, 32
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	for i := range aData {
		aData[i] = float32(i%7+1) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i%5+1) * 0.1
	}

	// Compute expected naively
	expected := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += aData[i*k+l] * bData[l*n+j]
			}
			expected[i*n+j] = sum
		}
	}

	g := NewGraph()
	a := NewMatrix(g, Float32, WithShape(m, k), WithName("a"), WithValue(
		tensor.New(tensor.WithShape(m, k), tensor.WithBacking(aData)),
	))
	b := NewMatrix(g, Float32, WithShape(k, n), WithName("b"), WithValue(
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

	result := c.Value().Data().([]float32)
	for i := range result {
		tol := float64(1e-3) * math.Max(1.0, math.Abs(float64(expected[i])))
		if math.Abs(float64(result[i]-expected[i])) > tol {
			t.Errorf("result[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestNEON_Conv2d(t *testing.T) {
	impl := NEONImplementation()
	if impl == nil {
		t.Skip("NEON not available on this platform")
	}

	oldBLAS := WhichBLAS()
	Use(impl)
	defer Use(oldBLAS)

	g := NewGraph()
	x := NodeFromAny(g, tensor.New(
		tensor.WithShape(1, 1, 7, 5),
		tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34})))
	filter := NodeFromAny(g, tensor.New(
		tensor.WithShape(1, 1, 3, 3),
		tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1})))
	y := Must(Conv2d(x, filter, []int{3, 3}, []int{0, 0}, []int{2, 2}, []int{1, 1}))
	m := NewTapeMachine(g)
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	output := y.Value().Data().([]float32)
	expected := []float32{54, 72, 144, 162, 234, 252}
	for i := range expected {
		if math.Abs(float64(output[i]-expected[i])) > 1e-3 {
			t.Fatalf("output[%d] = %f, want %f", i, output[i], expected[i])
		}
	}
}

func BenchmarkNEON_MatMul_F32(b *testing.B) {
	impl := NEONImplementation()
	if impl == nil {
		b.Skip("NEON not available on this platform")
	}

	oldBLAS := WhichBLAS()
	Use(impl)
	defer Use(oldBLAS)

	sizes := []int{32, 64, 128, 256}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			g := NewGraph()
			aBack := tensor.Random(Float32, n*n)
			bBack := tensor.Random(Float32, n*n)
			aNode := NewMatrix(g, Float32, WithShape(n, n), WithName("a"), WithValue(
				tensor.New(tensor.WithShape(n, n), tensor.WithBacking(aBack)),
			))
			bNode := NewMatrix(g, Float32, WithShape(n, n), WithName("b"), WithValue(
				tensor.New(tensor.WithShape(n, n), tensor.WithBacking(bBack)),
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
