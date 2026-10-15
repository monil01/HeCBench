# IRIS DMem GEMM suite

The four CLI arguments, deterministic inputs, row-major A*B, alpha=1,
beta=0, nine cases, and two default make-run sizes follow the CUDA/HIP
source. The source compute-mode switch falls through to ordinary FP32;
the four float labels therefore use the same FP32 operation here too.

Portability compromise P1: vendor dispatch is replaced by one-output-per-thread
HIP kernels submitted through IRIS DMem. Half storage uses FP32 accumulation
in both cases, including the native compute-16 case; that case is numerically
equivalent within the documented storage-precision tolerance, not bitwise
identical. There is no vendor algorithm-selection or tensor-core performance
equivalence claim. Timings include IRIS task overhead and are not hipBLAS
kernel performance. No CPU fallback computes the benchmark output.

Every output is checked against a separate long-double host dot product using
the actual quantized input values, with FP64 1e-10, integer exact, other
storage types atol=rtol=0.008. IRIS_GEMM_DUMP=1 prints every value for
verify.py; local evidence compares all nine native modes, not only the ten
samples printed by the original program. Bounded 32x32x32 and 32x48x64
cases are validation cases, not claims of default 4096/8192-size completion.
