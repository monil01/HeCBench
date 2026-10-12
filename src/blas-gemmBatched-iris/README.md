# blas-gemmBatched IRIS Port

This port keeps the CUDA/HIP benchmark initialization, command-line flags, and
double-precision host reference check. The native HIP baseline uses hipBLAS
batched GEMM; the IRIS version launches a dmem-backed HIP kernel for the
matrix-vector case used by the benchmark.
