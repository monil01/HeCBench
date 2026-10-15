# IRIS DMem Lt-equivalent GEMM

The seven default cases and optional FP64_GEMM case preserve CLI, input
quantization, row-major A*B, alpha=1/beta=0, and original default make-run
sizes. Both source compute-mode switches fall through to ordinary FP32.
All GPU allocations, data movement and dispatch are IRIS DMem/tasks.

P1: HIP dot-product kernels replace opaque vendor dispatch, not its algorithm
selection or performance. FP16/BF16 accumulate in float; integer is exact.
Every output is checked independently with long-double dots from quantized
inputs (atol=rtol=0.008; optional double 1e-10, integer exact).

Baseline repair: both original Lt sources swap A/B in test_gemm and swap
again at the actual Matmul call. For unequal M/N, descriptor sizes can exceed
the pointed allocation. The port implements the advertised A*B. Validation
uses an isolated native source copy changing only the second swap, plus
whole-output diagnostics; untouched CUDA/HIP files and snapshots are retained.
baseline_fix.patch documents the one-line semantic repair. Original square
native run also succeeded because A and B have identical initialization.
No claim of agreement with undefined out-of-bounds nonsquare native behavior
is made. Native library calls are real hipBLASLt on gfx908, not emulation.

Validated 32x48x64 repeat2 and 32x32x32 repeat2; production 4096/8192 defaults
are retained but not validated. IRIS_GEMM_DUMP=1 emits every output.
