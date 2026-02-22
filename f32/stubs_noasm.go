//go:build !arm64

package f32

// AxpyUnitary computes y[i] += alpha * x[i] for all i.
func AxpyUnitary(alpha float32, x, y []float32) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

// DotUnitary computes the dot product sum(x[i]*y[i]).
func DotUnitary(x, y []float32) float32 {
	var sum float32
	for i, v := range x {
		sum += v * y[i]
	}
	return sum
}

// Sum computes the sum of all elements in x.
func Sum(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v
	}
	return sum
}
