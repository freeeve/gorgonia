package f32

import "gonum.org/v1/gonum/blas"

// Sgemv performs single-precision general matrix-vector multiplication.
// y = alpha * op(A) * x + beta * y
func Sgemv(tA blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// Handle beta scaling of y
	var lenY int
	if tA == blas.NoTrans {
		lenY = m
	} else {
		lenY = n
	}

	if beta != 1 {
		if incY == 1 {
			if beta == 0 {
				for i := 0; i < lenY; i++ {
					y[i] = 0
				}
			} else {
				for i := 0; i < lenY; i++ {
					y[i] *= beta
				}
			}
		} else {
			if beta == 0 {
				for i := 0; i < lenY; i++ {
					y[i*incY] = 0
				}
			} else {
				for i := 0; i < lenY; i++ {
					y[i*incY] *= beta
				}
			}
		}
	}

	if alpha == 0 {
		return
	}

	if tA == blas.NoTrans {
		sgemvN(m, n, alpha, a, lda, x, incX, y, incY)
	} else {
		sgemvT(m, n, alpha, a, lda, x, incX, y, incY)
	}
}

// sgemvN computes y += alpha * A * x (no transpose).
func sgemvN(m, n int, alpha float32, a []float32, lda int, x []float32, incX int, y []float32, incY int) {
	if incX == 1 && incY == 1 {
		for i := 0; i < m; i++ {
			y[i] += alpha * DotUnitary(a[i*lda:i*lda+n], x[:n])
		}
		return
	}
	for i := 0; i < m; i++ {
		var sum float32
		ix := 0
		for j := 0; j < n; j++ {
			sum += a[i*lda+j] * x[ix]
			ix += incX
		}
		y[i*incY] += alpha * sum
	}
}

// sgemvT computes y += alpha * A^T * x (transpose).
func sgemvT(m, n int, alpha float32, a []float32, lda int, x []float32, incX int, y []float32, incY int) {
	if incX == 1 && incY == 1 {
		for i := 0; i < m; i++ {
			tmp := alpha * x[i]
			AxpyUnitary(tmp, a[i*lda:i*lda+n], y[:n])
		}
		return
	}
	ix := 0
	for i := 0; i < m; i++ {
		tmp := alpha * x[ix]
		jy := 0
		for j := 0; j < n; j++ {
			y[jy] += tmp * a[i*lda+j]
			jy += incY
		}
		ix += incX
	}
}
