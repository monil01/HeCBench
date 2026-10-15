# CUDA-semantic NVFP4 IRIS port

Preserves six original shapes, repeat CLI/default100, one warmup, alpha1/beta0,
packed E2M1 0x22 inputs, UE4M3 scale0x38 per16, column-major BF16 output,
and the original whole-output 1e-2 relative check. Default inputs have no RNG.
`HECBENCH_IRIS_SHAPE=17,29,256 make run RUN_ARGS=2` bounds validation without
changing defaults. `HECBENCH_IRIS_PATTERNED=1` is an extra nonuniform fixture:
all finite E2M1 codes, signs, subnormals and several scales are exercised, and
every BF16 output is compared against independent double-precision dot products.
Scale tensors use explicit logical row/block indexing in this software path;
no vendor swizzled-layout parity is claimed for the extra patterned fixture.

Scope-expanded P1 adaptation: HIP decode tasks expand the packed/scaled values
to FP32 DMem tensors, an IRIS host2hip bridge task calls real hipBLAS SGEMM on
those DMem GPU pointers and the IRIS stream, then a HIP task rounds to BF16.
No native allocation/copy/launch and no CPU fallback for benchmark math.
This implements the full mathematical source operation, NOT hardware FP4 GEMM;
memory use, accumulation and performance differ and no FP4 TFLOPS are reported.

Important original-source difference: the HIP sibling implements MXFP4 with
UE8M0 scales per32, not CUDA's NVFP4 with UE4M3 scales per16. The CUDA meaning
is retained here. Native MI100/gfx908 lacks the original HIP block-scaled
operation; SKIP is never accepted as native numerical parity. Independent
whole-output evidence is recorded instead, without fabricated native runtime.
The full six production sizes are preserved but not claimed validated locally.
