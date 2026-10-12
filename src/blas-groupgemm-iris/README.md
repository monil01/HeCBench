# blas-groupgemm IRIS Port

The HIP baseline uses hipBLASLt grouped GEMM. The IRIS port preserves the same
MoE grouped-GEMM input generation and host reference verification, but flattens
all groups into dmem buffers and launches a HIP kernel for half-precision
column-major GEMM per group.
