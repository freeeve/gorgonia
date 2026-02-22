package gorgonia

import (
	"gorgonia.org/tensor"
)

// optimizedReduceMiddleAxis performs an optimized reduction for single
// middle-axis sum reductions by computing the result with a tiled loop
// that improves cache locality compared to the default strided access.
//
// This only applies when:
// - opName is "sum"
// - along has exactly 1 axis
// - the axis is not 0 (already fast) and not the last axis (already fast)
// - total elements >= 4096 (overhead dominates for small tensors)
// - the stride along the reduction axis causes cache misses
// - the data type is float32 or float64
func optimizedReduceMiddleAxis(t *tensor.Dense, opName string, along []int) (*tensor.Dense, bool) {
	if opName != "sum" || len(along) != 1 {
		return nil, false
	}

	axis := along[0]
	ndim := t.Dims()

	if axis == 0 || axis == ndim-1 {
		return nil, false
	}

	totalSize := t.Shape().TotalSize()
	if totalSize < 4096 {
		return nil, false
	}

	strides := t.Strides()
	dtypeSize := int(t.Dtype().Size())
	strideBytes := strides[axis] * dtypeSize
	if strideBytes < 4096 {
		return nil, false
	}

	shape := t.Shape()

	// Flatten the shape into [outer, reduceN, inner] for the tiled kernel.
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}
	reduceN := shape[axis]
	inner := 1
	for i := axis + 1; i < ndim; i++ {
		inner *= shape[i]
	}

	outShape := make([]int, 0, ndim-1)
	for i := 0; i < ndim; i++ {
		if i != axis {
			outShape = append(outShape, shape[i])
		}
	}

	switch data := t.Data().(type) {
	case []float32:
		out := make([]float32, outer*inner)
		reduceMiddleSumF32(data, out, outer, reduceN, inner)
		return tensor.New(tensor.WithShape(outShape...), tensor.WithBacking(out)), true

	case []float64:
		out := make([]float64, outer*inner)
		reduceMiddleSumF64(data, out, outer, reduceN, inner)
		return tensor.New(tensor.WithShape(outShape...), tensor.WithBacking(out)), true
	}

	return nil, false
}

// reduceMiddleSumF32 sums along the middle axis of a logically shaped
// [outer, reduceN, inner] float32 array, writing the [outer, inner] result.
// It tiles the inner dimension to improve cache utilization.
func reduceMiddleSumF32(data []float32, out []float32, outer, reduceN, inner int) {
	const tileSize = 256
	outerStride := reduceN * inner

	for o := 0; o < outer; o++ {
		oBase := o * outerStride
		oOut := o * inner
		for iStart := 0; iStart < inner; iStart += tileSize {
			iEnd := iStart + tileSize
			if iEnd > inner {
				iEnd = inner
			}
			tileLen := iEnd - iStart
			for r := 0; r < reduceN; r++ {
				rBase := oBase + r*inner + iStart
				oIdx := oOut + iStart
				for i := 0; i < tileLen; i++ {
					out[oIdx+i] += data[rBase+i]
				}
			}
		}
	}
}

// reduceMiddleSumF64 sums along the middle axis of a logically shaped
// [outer, reduceN, inner] float64 array, writing the [outer, inner] result.
func reduceMiddleSumF64(data []float64, out []float64, outer, reduceN, inner int) {
	const tileSize = 256
	outerStride := reduceN * inner

	for o := 0; o < outer; o++ {
		oBase := o * outerStride
		oOut := o * inner
		for iStart := 0; iStart < inner; iStart += tileSize {
			iEnd := iStart + tileSize
			if iEnd > inner {
				iEnd = inner
			}
			tileLen := iEnd - iStart
			for r := 0; r < reduceN; r++ {
				rBase := oBase + r*inner + iStart
				oIdx := oOut + iStart
				for i := 0; i < tileLen; i++ {
					out[oIdx+i] += data[rBase+i]
				}
			}
		}
	}
}
