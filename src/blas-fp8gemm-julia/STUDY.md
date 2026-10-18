# FP8 GEMM: measured native operation blocker

The unchanged native source compiles for sm70. Its cuBLASLt operation requires FP8 E4M3 A/B, BF16 C/D, Float32 computation, per-tensor A/B scales, transposed A, non-transposed B, fast accumulation and a4MiB workspace. `native_capability_probe.cu` queries that descriptor contract at all six original shapes and an aligned64-cubed case without executing matmul or fabricating timing.

On GPU6 Tesla V100sm70, both system cuBLASLt12.0 (version120002) and isolated HPCSDK12.9 (version120900) return status15 CUBLAS_STATUS_NOT_SUPPORTED and zero algorithms for every shape. Exact commands, scoped include/library/RPATH paths and hashed logs are listed in `diagnostic_evidence.json`; complete raw logs stay in local STATE. No shared Julia preferences changed.

Historical `main.jl` merely launches an Int32 marker then prints hardcoded zero times, dimensions and PASS. This is S1, not numerical verification of GEMM. KA/JACC translation remains unattempted; no framework unsupported-feature conclusion, no blanket BF16 restriction and no FP32 emulation is claimed.

Reproduce the probe with `nvcc -O3 -std=c++17 -arch=sm_70 native_capability_probe.cu -lcublasLt -o probe` from this folder; run only through the shared GPU6 flock protocol. The pinned12.9 command and linkage are in the diagnostic record. Resume on a native-supported GPU/library configuration, repair the genuine CUDA.jl reference, then KA followed byJACC.

## Final available-environment review

limitation_review.json verifies all archived source/log hashes and both seven-shape capability outcomes. This is a reviewed unavailable native prerequisite on the availableV100/library configurations, not a KA/JACC feature failure. No output/time exists; historical CUDAjlmarker remains unauthenticated scientific computation. Resume requires a native-supported GPU/library contract.
