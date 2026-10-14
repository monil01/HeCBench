#include <hip/hip_runtime.h>
// Export original kernel names for IRIS string lookup. Device helpers and
// numerical kernels remain the native HIP translations of the CUDA routines.
#undef __global__
#define __global__ extern "C" __attribute__((global))
#include "../xlqc-hip/cuda_rys_sp.cu"
#include "../xlqc-hip/cuda_rys_dp.cu"
