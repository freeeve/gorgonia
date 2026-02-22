#include "textflag.h"

// func AxpyUnitary(alpha float32, x, y []float32)
//
// Stack layout (arm64, 8-byte aligned):
//   alpha  float32  +0(FP)   (padded to 8 bytes)
//   x_base ptr      +8(FP)
//   x_len  int      +16(FP)
//   x_cap  int      +24(FP)
//   y_base ptr      +32(FP)
//   y_len  int      +40(FP)
//   y_cap  int      +48(FP)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	FMOVS	alpha+0(FP), F0       // F0 = alpha
	MOVD	x_base+8(FP), R0      // R0 = &x[0]
	MOVD	x_len+16(FP), R1      // R1 = len(x)
	MOVD	y_base+32(FP), R2     // R2 = &y[0]

	CBZ	R1, done               // if len == 0, return

	// Broadcast alpha to all 4 lanes of V0
	VDUP	V0.S[0], V0.S4

	// Check if we have at least 16 elements for the vectorized loop
	CMP	$16, R1
	BLT	tail_start

loop16:
	// Load 16 elements from x (4 vectors of 4 floats)
	VLD1.P	16(R0), [V1.S4]
	VLD1.P	16(R0), [V2.S4]
	VLD1.P	16(R0), [V3.S4]
	VLD1.P	16(R0), [V4.S4]

	// Load 16 elements from y
	VLD1.P	16(R2), [V5.S4]
	VLD1.P	16(R2), [V6.S4]
	VLD1.P	16(R2), [V7.S4]
	VLD1.P	16(R2), [V8.S4]

	// y[i] += alpha * x[i] using fused multiply-add
	VFMLA	V0.S4, V1.S4, V5.S4
	VFMLA	V0.S4, V2.S4, V6.S4
	VFMLA	V0.S4, V3.S4, V7.S4
	VFMLA	V0.S4, V4.S4, V8.S4

	// Store 16 results back to y (rewind pointer, then store with post-increment)
	SUB	$64, R2, R2
	VST1.P	[V5.S4], 16(R2)
	VST1.P	[V6.S4], 16(R2)
	VST1.P	[V7.S4], 16(R2)
	VST1.P	[V8.S4], 16(R2)

	SUB	$16, R1, R1
	CMP	$16, R1
	BGE	loop16

tail_start:
	CBZ	R1, done

tail:
	FMOVS	(R0), F1               // load x[i]
	FMOVS	(R2), F2               // load y[i]
	FMULS	F0, F1                 // F1 = alpha * x[i]  (scalar F0 still has alpha)
	FADDS	F2, F1                 // F1 = y[i] + alpha * x[i]
	FMOVS	F1, (R2)               // store y[i]
	ADD	$4, R0
	ADD	$4, R2
	SUB	$1, R1, R1
	CBNZ	R1, tail

done:
	RET
