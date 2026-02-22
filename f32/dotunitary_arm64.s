#include "textflag.h"

// func DotUnitary(x, y []float32) float32
//
// Stack layout (arm64, 8-byte aligned):
//   x_base ptr      +0(FP)
//   x_len  int      +8(FP)
//   x_cap  int      +16(FP)
//   y_base ptr      +24(FP)
//   y_len  int      +32(FP)
//   y_cap  int      +40(FP)
//   ret    float32  +48(FP)
TEXT ·DotUnitary(SB), NOSPLIT, $0
	MOVD	x_base+0(FP), R0      // R0 = &x[0]
	MOVD	x_len+8(FP), R1       // R1 = len(x)
	MOVD	y_base+24(FP), R2     // R2 = &y[0]

	// Zero accumulators
	VEOR	V16.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16
	VEOR	V18.B16, V18.B16, V18.B16
	VEOR	V19.B16, V19.B16, V19.B16

	CBZ	R1, reduce

	CMP	$16, R1
	BLT	tail_start

loop16:
	// Load 16 elements from x
	VLD1.P	16(R0), [V0.S4]
	VLD1.P	16(R0), [V1.S4]
	VLD1.P	16(R0), [V2.S4]
	VLD1.P	16(R0), [V3.S4]

	// Load 16 elements from y
	VLD1.P	16(R2), [V4.S4]
	VLD1.P	16(R2), [V5.S4]
	VLD1.P	16(R2), [V6.S4]
	VLD1.P	16(R2), [V7.S4]

	// Accumulate x[i]*y[i]
	VFMLA	V0.S4, V4.S4, V16.S4
	VFMLA	V1.S4, V5.S4, V17.S4
	VFMLA	V2.S4, V6.S4, V18.S4
	VFMLA	V3.S4, V7.S4, V19.S4

	SUB	$16, R1, R1
	CMP	$16, R1
	BGE	loop16

tail_start:
	CBZ	R1, reduce

	// Scalar accumulator F4 = 0
	FMOVS	ZR, F4

tail:
	FMOVS	(R0), F0
	FMOVS	(R2), F1
	FMADDS	F0, F4, F1, F4        // F4 = F1 * F0 + F4
	ADD	$4, R0
	ADD	$4, R2
	SUB	$1, R1, R1
	CBNZ	R1, tail

	// Add scalar tail sum into V16 lane 0
	VMOV	V16.S[0], V20.S[0]    // F20 = V16.S[0]
	FADDS	F20, F4, F4           // F4 = V16.S[0] + tail_sum
	VMOV	V4.S[0], V16.S[0]    // V16.S[0] = F4

reduce:
	// Combine the 4 accumulator vectors (float vector add)
	// fadd v16.4s, v16.4s, v17.4s
	WORD	$0x4E31D610
	// fadd v16.4s, v16.4s, v18.4s
	WORD	$0x4E32D610
	// fadd v16.4s, v16.4s, v19.4s
	WORD	$0x4E33D610

	// Horizontal sum of V16.S4 = [a, b, c, d]
	VMOV	V16.S[0], V20.S[0]    // F20 = a
	VMOV	V16.S[1], V21.S[0]    // F21 = b
	VMOV	V16.S[2], V22.S[0]    // F22 = c
	VMOV	V16.S[3], V23.S[0]    // F23 = d

	FADDS	F20, F21, F20          // F20 = a + b
	FADDS	F22, F23, F22          // F22 = c + d
	FADDS	F20, F22, F20          // F20 = a + b + c + d

	FMOVS	F20, ret+48(FP)
	RET
