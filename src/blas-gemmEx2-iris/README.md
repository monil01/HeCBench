# blas-gemmEx2 IRIS status

This benchmark is deferred for the IRIS dmem-only pass.

The CUDA/HIP source is a cuBLASLt/hipBLASLt benchmark. It allocates typed
matrices and asks the vendor BLASLt library to select and run matmul algorithms
for fp32 compute modes, fp16, bfloat16, and int8. There is no benchmark-owned
CUDA kernel body to translate into an IRIS task, and using hipBLASLt directly
from the IRIS host path would violate the current dmem-only porting rule.

A faithful port needs either an IRIS-supported BLASLt task binding or a
separately scoped benchmark-owned GEMM implementation for each tested type and
compute mode. Neither is present in this repository.
