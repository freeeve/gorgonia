package f32

import (
	"math"
	"testing"
)

func TestAxpyUnitary(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 15, 16, 17, 31, 32, 33, 100, 1000} {
		x := make([]float32, n)
		y := make([]float32, n)
		yExpected := make([]float32, n)
		for i := range x {
			x[i] = float32(i + 1)
			y[i] = float32(i) * 0.5
			yExpected[i] = y[i] + 2.0*x[i]
		}
		AxpyUnitary(2.0, x, y)
		for i := range y {
			if math.Abs(float64(y[i]-yExpected[i])) > 1e-5 {
				t.Fatalf("n=%d: y[%d] = %f, want %f", n, i, y[i], yExpected[i])
			}
		}
	}
}

func TestAxpyUnitary_AlphaZero(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	y := []float32{10, 20, 30, 40}
	expected := []float32{10, 20, 30, 40}
	AxpyUnitary(0.0, x, y)
	for i := range y {
		if y[i] != expected[i] {
			t.Fatalf("y[%d] = %f, want %f", i, y[i], expected[i])
		}
	}
}

func TestAxpyUnitary_NegativeAlpha(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	y := []float32{10, 20, 30, 40}
	expected := []float32{9, 18, 27, 36}
	AxpyUnitary(-1.0, x, y)
	for i := range y {
		if math.Abs(float64(y[i]-expected[i])) > 1e-5 {
			t.Fatalf("y[%d] = %f, want %f", i, y[i], expected[i])
		}
	}
}

func TestDotUnitary(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 15, 16, 17, 31, 32, 33, 100, 1000} {
		x := make([]float32, n)
		y := make([]float32, n)
		var expected float64
		for i := range x {
			x[i] = float32(i + 1)
			y[i] = float32(i)*0.5 + 1
			expected += float64(x[i]) * float64(y[i])
		}
		result := DotUnitary(x, y)
		// Use relative tolerance for larger values
		tol := 1e-4 * math.Max(1.0, math.Abs(expected))
		if math.Abs(float64(result)-expected) > tol {
			t.Fatalf("n=%d: got %f, want %f (diff=%e)", n, result, expected, math.Abs(float64(result)-expected))
		}
	}
}

func TestDotUnitary_Orthogonal(t *testing.T) {
	x := []float32{1, 0, 0, 0}
	y := []float32{0, 1, 0, 0}
	result := DotUnitary(x, y)
	if result != 0 {
		t.Fatalf("orthogonal dot: got %f, want 0", result)
	}
}

func TestSum(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 15, 16, 17, 31, 32, 33, 100, 1000} {
		x := make([]float32, n)
		var expected float64
		for i := range x {
			x[i] = float32(i + 1)
			expected += float64(x[i])
		}
		result := Sum(x)
		tol := 1e-4 * math.Max(1.0, math.Abs(expected))
		if math.Abs(float64(result)-expected) > tol {
			t.Fatalf("n=%d: got %f, want %f (diff=%e)", n, result, expected, math.Abs(float64(result)-expected))
		}
	}
}

func TestSum_Empty(t *testing.T) {
	result := Sum(nil)
	if result != 0 {
		t.Fatalf("Sum(nil) = %f, want 0", result)
	}
}

func BenchmarkAxpyUnitary(b *testing.B) {
	for _, n := range []int{16, 64, 256, 1024, 4096} {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(i)
			y[i] = float32(i)
		}
		b.Run(toString(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				AxpyUnitary(2.0, x, y)
			}
		})
	}
}

func BenchmarkDotUnitary(b *testing.B) {
	for _, n := range []int{16, 64, 256, 1024, 4096} {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(i)
			y[i] = float32(i)
		}
		b.Run(toString(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				DotUnitary(x, y)
			}
		})
	}
}

func BenchmarkSum(b *testing.B) {
	for _, n := range []int{16, 64, 256, 1024, 4096} {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(i)
		}
		b.Run(toString(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Sum(x)
			}
		})
	}
}

func toString(n int) string {
	switch {
	case n >= 1024:
		return string(rune('0'+n/1024)) + "k"
	default:
		s := ""
		v := n
		for v > 0 {
			s = string(rune('0'+v%10)) + s
			v /= 10
		}
		if s == "" {
			return "0"
		}
		return s
	}
}
