//go:build arm64

package gorgonia

import (
	"gonum.org/v1/gonum/blas"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
	"gorgonia.org/gorgonia/f32"
)

// neonBLAS embeds gonum's Implementation, overriding f32 BLAS operations
// with ARM64 NEON-optimized versions.
type neonBLAS struct {
	gonumblas.Implementation
}

// NEONImplementation returns a BLAS implementation that uses ARM64 NEON
// for float32 operations, delegating everything else to gonum.
func NEONImplementation() BLAS {
	return neonBLAS{}
}

func (n neonBLAS) Sgemm(tA, tB blas.Transpose, m, nCols, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	f32.Sgemm(tA, tB, m, nCols, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (n neonBLAS) Sgemv(tA blas.Transpose, m, nCols int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	f32.Sgemv(tA, m, nCols, alpha, a, lda, x, incX, beta, y, incY)
}
