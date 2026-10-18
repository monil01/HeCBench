# MXFP6 GEMM: scoped unavailable native prerequisite

## Scientific operation

The native source requests block-scaled row-pair matrix multiplication: packed FP6 E2M3 A/B (four6-bit codes per3bytes), UE8M0 scales per32K elements, FP32 compute, BF16 C/D, TN descriptors and32MiB workspace. Original M=N=8192 and K=512/1024/2048/4096/8192. The supplied fixture uses uniformFP6code0x08 andscale127 (ones); it does not establish arbitrary quantized-ML application quality.

## Actual capability evidence

Unchanged source builds with both available compilers. Its CUDA12.0 executable skips for toolkit<12.8; its isolated CUDA12.9 executable skips V100sm70 at the host architecture guard. Both return0 but execute no GEMM. A separate diagnostic removes no device algorithm: it queries the exact source descriptors without executing matmul. All five original shapes and an aligned128³ case return cuBLASLt status15 (NOT_SUPPORTED). The returned-count output remains its initialized−1 sentinel on error; **algorithm count is unknown**, not a negative count or measured zero. Actual dlsym/dladdr resolves cuBLASLt12.9.0.13, runtime-reported120900; resolved bytes/hash and all commands/log hashes are in `limitation_review.json`. No shared Julia preferences changed.

No numerical output or execution time exists. This is a reviewed device/library/operation prerequisite on V100 plus available12.0/12.9, not a general assertion that FP6 arithmetic or KA/JACC is unsupported. Source comments alone did not establish the blocker: actual guarded native executions and an independent descriptor query were retained. No repeated unchanged diagnosis is required.

## Historical reference, costs and resumption

Historical `main.jl` includes mcpr, whose actual source computes Gaussian-utility categorical-choice frequencies, not GEMM. This is S1; the alias is preserved and was not executed. No CUDA.jl/KA/JACC target candidate or device JIT was attempted. Two unchanged-native builds, one descriptor-probe build and three diagnostic executions are reference/capability effort, not translation successes. Prospective reference/workflow usage windows are contextual/cache-inclusive; unknown historical translation cost remains unknown.

Resume on a device/cuBLASLt combination accepting and executing the exact native descriptor; then restore genuine CUDA.jl, verify KA and finally JACC. Software packed-decode arithmetic elsewhere cannot by itself provide an unchanged-native four-way reference for this unavailable library operation. CPU/GPU output, scientific validation and performance remain unknown.
