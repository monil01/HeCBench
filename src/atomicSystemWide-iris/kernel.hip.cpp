#include <hip/hip_runtime.h>

extern "C" __global__ void atomicKernel(int *atom_arr, const int loop_num)
{
  const int tid = static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x);

  for (int i = 0; i < loop_num; ++i) {
    atomicAdd_system(&atom_arr[0], 10);
    atomicExch_system(&atom_arr[1], tid);
    atomicMax_system(&atom_arr[2], tid);
    atomicMin_system(&atom_arr[3], tid);
    atomicCAS_system(&atom_arr[6], tid - 1, tid);
    atomicAnd_system(&atom_arr[7], 2 * tid + 7);
    atomicOr_system(&atom_arr[8], 1 << tid);
    atomicXor_system(&atom_arr[9], tid);
  }
}
