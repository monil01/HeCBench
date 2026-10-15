#include <hip/hip_runtime.h>
// Exact original initializer: same 256-thread tiling, row-major enumeration
// into column-major padded storage, and 499-prime stride rule.
extern "C" __global__ void set_input_matrix_kernel(size_t nr,size_t nc,size_t nru,float* d,size_t base,float value) {
  const size_t index=threadIdx.x+blockDim.x*blockIdx.x;
  if(index>=nr*nc)return;
  const size_t r=index%nr,c=index/nr;
  const size_t stride=1+(r+base)%499;
  d[r+nru*c]=c%stride?0.0f:value;
}
