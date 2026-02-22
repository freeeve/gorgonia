package f32

import "gonum.org/v1/gonum/blas"

const blockM = 64
const blockN = 64
const blockK = 64

// Sgemm performs single-precision general matrix multiplication.
// C = alpha * op(A) * op(B) + beta * C
// where op(X) is X or X^T depending on the transpose parameter.
func Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 {
		return
	}

	// Scale C by beta
	if beta != 1 {
		if beta == 0 {
			for i := 0; i < m; i++ {
				ci := c[i*ldc : i*ldc+n]
				for j := range ci {
					ci[j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				ci := c[i*ldc : i*ldc+n]
				for j := range ci {
					ci[j] *= beta
				}
			}
		}
	}

	if alpha == 0 || k == 0 {
		return
	}

	switch {
	case tA == blas.NoTrans && tB == blas.NoTrans:
		sgemmNN(m, n, k, alpha, a, lda, b, ldb, c, ldc)
	case tA == blas.Trans && tB == blas.NoTrans:
		sgemmTN(m, n, k, alpha, a, lda, b, ldb, c, ldc)
	case tA == blas.NoTrans && tB == blas.Trans:
		sgemmNT(m, n, k, alpha, a, lda, b, ldb, c, ldc)
	case tA == blas.Trans && tB == blas.Trans:
		sgemmTT(m, n, k, alpha, a, lda, b, ldb, c, ldc)
	}
}

// sgemmNN computes C += alpha * A * B (no transpose).
func sgemmNN(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for kk := 0; kk < k; kk += blockK {
		kEnd := kk + blockK
		if kEnd > k {
			kEnd = k
		}
		for ii := 0; ii < m; ii += blockM {
			iEnd := ii + blockM
			if iEnd > m {
				iEnd = m
			}
			for jj := 0; jj < n; jj += blockN {
				jEnd := jj + blockN
				if jEnd > n {
					jEnd = n
				}
				nBlock := jEnd - jj
				for i := ii; i < iEnd; i++ {
					for j := kk; j < kEnd; j++ {
						tmp := alpha * a[i*lda+j]
						AxpyUnitary(tmp, b[j*ldb+jj:j*ldb+jj+nBlock], c[i*ldc+jj:i*ldc+jj+nBlock])
					}
				}
			}
		}
	}
}

// sgemmTN computes C += alpha * A^T * B.
func sgemmTN(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for kk := 0; kk < k; kk += blockK {
		kEnd := kk + blockK
		if kEnd > k {
			kEnd = k
		}
		for ii := 0; ii < m; ii += blockM {
			iEnd := ii + blockM
			if iEnd > m {
				iEnd = m
			}
			for jj := 0; jj < n; jj += blockN {
				jEnd := jj + blockN
				if jEnd > n {
					jEnd = n
				}
				nBlock := jEnd - jj
				for i := ii; i < iEnd; i++ {
					for j := kk; j < kEnd; j++ {
						// A is transposed: A^T[i][j] = A[j][i]
						tmp := alpha * a[j*lda+i]
						AxpyUnitary(tmp, b[j*ldb+jj:j*ldb+jj+nBlock], c[i*ldc+jj:i*ldc+jj+nBlock])
					}
				}
			}
		}
	}
}

// sgemmNT computes C += alpha * A * B^T.
func sgemmNT(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for ii := 0; ii < m; ii += blockM {
		iEnd := ii + blockM
		if iEnd > m {
			iEnd = m
		}
		for jj := 0; jj < n; jj += blockN {
			jEnd := jj + blockN
			if jEnd > n {
				jEnd = n
			}
			for kk := 0; kk < k; kk += blockK {
				kEnd := kk + blockK
				if kEnd > k {
					kEnd = k
				}
				kBlock := kEnd - kk
				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j++ {
						// B^T[j][kk:kEnd] = B[j*ldb+kk : j*ldb+kEnd]
						c[i*ldc+j] += alpha * DotUnitary(a[i*lda+kk:i*lda+kk+kBlock], b[j*ldb+kk:j*ldb+kk+kBlock])
					}
				}
			}
		}
	}
}

// sgemmTT computes C += alpha * A^T * B^T.
func sgemmTT(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for ii := 0; ii < m; ii += blockM {
		iEnd := ii + blockM
		if iEnd > m {
			iEnd = m
		}
		for jj := 0; jj < n; jj += blockN {
			jEnd := jj + blockN
			if jEnd > n {
				jEnd = n
			}
			for kk := 0; kk < k; kk += blockK {
				kEnd := kk + blockK
				if kEnd > k {
					kEnd = k
				}
				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j++ {
						var sum float32
						for l := kk; l < kEnd; l++ {
							sum += a[l*lda+i] * b[j*ldb+l]
						}
						c[i*ldc+j] += alpha * sum
					}
				}
			}
		}
	}
}
