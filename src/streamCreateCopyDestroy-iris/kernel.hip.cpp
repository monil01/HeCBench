#include <hip/hip_runtime.h>
// Untimed diagnostic kernel: read actual device copies, not cached host data.
extern "C" __global__ void read_copied_buffer(const float* input,float* output,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)output[i]=input[i];
}
