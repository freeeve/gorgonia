#include "textflag.h"

// func Sum(x []float32) float32
//
// Stack layout (arm64, 8-byte aligned):
//   x_base ptr      +0(FP)
//   x_len  int      +8(FP)
//   x_cap  int      +16(FP)
//   ret    float32  +24(FP)
TEXT ·Sum(SB), NOSPLIT, $0
	MOVD	x_base+0(FP), R0      // R0 = &x[0]
	MOVD	x_len+8(FP), R1       // R1 = len(x)

	// Zero accumulators
	VEOR	V16.B16, V16.B16, V16.B16
	VEOR	V17.B16, V17.B16, V17.B16
	VEOR	V18.B16, V18.B16, V18.B16
	VEOR	V19.B16, V19.B16, V19.B16

	CBZ	R1, reduce

	CMP	$16, R1
	BLT	tail_start

loop16:
	VLD1.P	16(R0), [V0.S4]
	VLD1.P	16(R0), [V1.S4]
	VLD1.P	16(R0), [V2.S4]
	VLD1.P	16(R0), [V3.S4]

	// fadd v16.4s, v16.4s, v0.4s  (float vector add)
	WORD	$0x4E20D610
	// fadd v17.4s, v17.4s, v1.4s
	WORD	$0x4E21D631
	// fadd v18.4s, v18.4s, v2.4s
	WORD	$0x4E22D652
	// fadd v19.4s, v19.4s, v3.4s
	WORD	$0x4E23D673

	SUB	$16, R1, R1
	CMP	$16, R1
	BGE	loop16

tail_start:
	CBZ	R1, reduce

	FMOVS	ZR, F4

tail:
	FMOVS	(R0), F0
	FADDS	F0, F4, F4
	ADD	$4, R0
	SUB	$1, R1, R1
	CBNZ	R1, tail

	// Add scalar tail sum into V16 lane 0
	VMOV	V16.S[0], V20.S[0]
	FADDS	F20, F4, F4
	VMOV	V4.S[0], V16.S[0]

reduce:
	// fadd v16.4s, v16.4s, v17.4s
	WORD	$0x4E31D610
	// fadd v16.4s, v16.4s, v18.4s
	WORD	$0x4E32D610
	// fadd v16.4s, v16.4s, v19.4s
	WORD	$0x4E33D610

	// Horizontal sum of V16.S4 = [a, b, c, d]
	VMOV	V16.S[0], V20.S[0]
	VMOV	V16.S[1], V21.S[0]
	VMOV	V16.S[2], V22.S[0]
	VMOV	V16.S[3], V23.S[0]

	FADDS	F20, F21, F20
	FADDS	F22, F23, F22
	FADDS	F20, F22, F20

	FMOVS	F20, ret+24(FP)
	RET
