# blas-gemmEx IRIS status

This benchmark is deferred for the IRIS dmem-only pass.

The CUDA/HIP source exercises `cublasGemmEx` / `hipblasGemmEx` across fp64,
fp32 compute modes, fp16, bfloat16, and int8. That behavior is library dispatch
rather than a benchmark-owned CUDA kernel body. The current porting rule for
this worker is IRIS dmem only, with no native HIP allocation/copy/launch or
vendor-library dispatch in the IRIS host path.

A faithful IRIS implementation would require either an IRIS-supported BLAS task
binding or a separately scoped benchmark-owned GEMM kernel suite for every
tested type and compute mode. Neither exists in this repository.
