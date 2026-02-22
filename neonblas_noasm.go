//go:build !arm64

package gorgonia

// NEONImplementation returns nil on non-ARM64 platforms.
// Use gonum's default Implementation instead.
func NEONImplementation() BLAS {
	return nil
}
