package f32

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/blas"
)

// naiveSgemm computes C = alpha * op(A) * op(B) + beta * C using naive triple loop.
func naiveSgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i*ldc+j] *= beta
		}
	}
	if alpha == 0 {
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				var aVal, bVal float32
				if tA == blas.NoTrans {
					aVal = a[i*lda+l]
				} else {
					aVal = a[l*lda+i]
				}
				if tB == blas.NoTrans {
					bVal = b[l*ldb+j]
				} else {
					bVal = b[j*ldb+l]
				}
				sum += aVal * bVal
			}
			c[i*ldc+j] += alpha * sum
		}
	}
}

func TestSgemm(t *testing.T) {
	cases := []struct {
		name    string
		tA      blas.Transpose
		tB      blas.Transpose
		m, n, k int
	}{
		{"NN_2x2x2", blas.NoTrans, blas.NoTrans, 2, 2, 2},
		{"NN_4x4x4", blas.NoTrans, blas.NoTrans, 4, 4, 4},
		{"NN_3x5x7", blas.NoTrans, blas.NoTrans, 3, 5, 7},
		{"NN_17x19x23", blas.NoTrans, blas.NoTrans, 17, 19, 23},
		{"NN_64x64x64", blas.NoTrans, blas.NoTrans, 64, 64, 64},
		{"NN_65x65x65", blas.NoTrans, blas.NoTrans, 65, 65, 65},
		{"NN_128x128x128", blas.NoTrans, blas.NoTrans, 128, 128, 128},
		{"TN_4x4x4", blas.Trans, blas.NoTrans, 4, 4, 4},
		{"TN_17x19x23", blas.Trans, blas.NoTrans, 17, 19, 23},
		{"NT_4x4x4", blas.NoTrans, blas.Trans, 4, 4, 4},
		{"NT_17x19x23", blas.NoTrans, blas.Trans, 17, 19, 23},
		{"TT_4x4x4", blas.Trans, blas.Trans, 4, 4, 4},
		{"TT_17x19x23", blas.Trans, blas.Trans, 17, 19, 23},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, n, k := tc.m, tc.n, tc.k
			var aRows, aCols, bRows, bCols int
			if tc.tA == blas.NoTrans {
				aRows, aCols = m, k
			} else {
				aRows, aCols = k, m
			}
			if tc.tB == blas.NoTrans {
				bRows, bCols = k, n
			} else {
				bRows, bCols = n, k
			}

			lda := aCols
			ldb := bCols
			ldc := n

			a := make([]float32, aRows*aCols)
			b := make([]float32, bRows*bCols)
			for i := range a {
				a[i] = float32(i%7+1) * 0.5
			}
			for i := range b {
				b[i] = float32(i%5+1) * 0.3
			}

			alpha := float32(1.5)
			beta := float32(0.5)

			// Expected result
			cExpected := make([]float32, m*n)
			for i := range cExpected {
				cExpected[i] = float32(i%3) * 0.1
			}
			cGot := make([]float32, m*n)
			copy(cGot, cExpected)

			naiveSgemm(tc.tA, tc.tB, m, n, k, alpha, a, lda, b, ldb, beta, cExpected, ldc)
			Sgemm(tc.tA, tc.tB, m, n, k, alpha, a, lda, b, ldb, beta, cGot, ldc)

			for i := range cExpected {
				tol := float64(1e-3) * math.Max(1.0, math.Abs(float64(cExpected[i])))
				if math.Abs(float64(cGot[i]-cExpected[i])) > tol {
					t.Fatalf("c[%d] = %f, want %f (diff=%e)", i, cGot[i], cExpected[i], math.Abs(float64(cGot[i]-cExpected[i])))
				}
			}
		})
	}
}

func TestSgemm_BetaZero(t *testing.T) {
	m, n, k := 4, 4, 4
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i + 1)
	}
	for i := range b {
		b[i] = float32(i + 1)
	}
	c := make([]float32, m*n)
	for i := range c {
		c[i] = 999 // should be zeroed by beta=0
	}
	Sgemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n)

	cExpected := make([]float32, m*n)
	naiveSgemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0, a, k, b, n, 0.0, cExpected, n)

	for i := range cExpected {
		if math.Abs(float64(c[i]-cExpected[i])) > 1e-3 {
			t.Fatalf("c[%d] = %f, want %f", i, c[i], cExpected[i])
		}
	}
}

func TestSgemv(t *testing.T) {
	m, n := 4, 5
	a := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i + 1)
	}
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i + 1)
	}
	y := make([]float32, m)

	Sgemv(blas.NoTrans, m, n, 1.0, a, n, x, 1, 0.0, y, 1)

	// Expected: row i of A dot x
	for i := 0; i < m; i++ {
		var expected float32
		for j := 0; j < n; j++ {
			expected += a[i*n+j] * x[j]
		}
		if math.Abs(float64(y[i]-expected)) > 1e-3 {
			t.Fatalf("y[%d] = %f, want %f", i, y[i], expected)
		}
	}
}

func TestSgemv_Trans(t *testing.T) {
	m, n := 4, 5
	a := make([]float32, m*n)
	for i := range a {
		a[i] = float32(i + 1)
	}
	x := make([]float32, m)
	for i := range x {
		x[i] = float32(i + 1)
	}
	y := make([]float32, n)

	Sgemv(blas.Trans, m, n, 1.0, a, n, x, 1, 0.0, y, 1)

	// Expected: col j of A dot x
	for j := 0; j < n; j++ {
		var expected float32
		for i := 0; i < m; i++ {
			expected += a[i*n+j] * x[i]
		}
		if math.Abs(float64(y[j]-expected)) > 1e-3 {
			t.Fatalf("y[%d] = %f, want %f", j, y[j], expected)
		}
	}
}

func BenchmarkSgemm(b *testing.B) {
	for _, n := range []int{64, 128, 256} {
		a := make([]float32, n*n)
		bm := make([]float32, n*n)
		c := make([]float32, n*n)
		for i := range a {
			a[i] = float32(i%7+1) * 0.5
			bm[i] = float32(i%5+1) * 0.3
		}
		b.Run(toString(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Sgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1.0, a, n, bm, n, 0.0, c, n)
			}
		})
	}
}
