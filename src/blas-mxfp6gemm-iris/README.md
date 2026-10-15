# MXFP6 E2M3 mathematical IRIS DMem port

Preserves original five shapes M=N=8192,K=512/1024/2048/4096/8192,
repeat CLI/default100, warmup, alpha1/beta0,
E2M3 code0x08 uniform1 input packed four values per three bytes, UE8M0 scale127
per32, column-major BF16 output, and original all-element rtol1e-2 check.

All device storage/movement uses DMem. HIP kernels unpack actual 6-bit codes
(including byte-straddling fields) and apply block scales into FP32 DMem
matrices. An IRIS bridge task calls real hipBLAS SGEMM on those GPU pointers
and the IRIS-owned stream, then HIP converts to BF16. No native benchmark
allocation/copy/launch and no CPU fallback. Scope-expanded P1: this is the full
mathematical MXFP6 operation, NOT hardware FP6 GEMM; storage, accumulation
and performance differ. No low-precision hardware TFLOPS claim is made.

`HECBENCH_IRIS_SHAPE=64,96,128 make run RUN_ARGS=2` is bounded validation only.
`HECBENCH_IRIS_PATTERNED=1 HECBENCH_IRIS_SHAPE=17,29,256 make run RUN_ARGS=3`
additionally exercises all64 finite codes/signs/subnormals and varying scales,
with each output independently checked using double dot products and BF16
exact-match counts. Scales use logical row/block indexing in this software
path; no vendor swizzled-layout parity for the extra fixture is claimed.

Native HIP block-scaled MXFP6 is not available on MI100/gfx908; the unchanged
native capability-probe outcome is logged separately, not a numerical PASS.
Default production shapes are retained but not claimed executed locally.

Decoder qualification: the original CUDA/HIP helper's exp==0 branch uses
`(mantissa/8)*0.5`; this exact source-helper convention is retained for the
extra patterned fixture. No all-code hardware FP6 decoder equivalence is
claimed without supported native hardware. The actual source uniform code0x08
is normal and unaffected by this subnormal convention.
