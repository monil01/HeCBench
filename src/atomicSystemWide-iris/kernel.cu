#include <cuda_runtime.h>

// MI100 cannot forward bitwise atomics to PCIe host DRAM. Preserve the
// indivisible operation with genuine SYSTEM-scope compare-and-swap.
__device__ int system_bitwise(int* address, int value, int operation) {
  int old = atomicCAS_system(address, 0, 0);
  for (;;) {
    const int expected = old;
    const int next = operation == 0 ? (expected & value) :
                     operation == 1 ? (expected | value) : (expected ^ value);
    old = atomicCAS_system(address, expected, next);
    if (old == expected) return old;
  }
}

__device__ void atomicBody(int *atom_arr, const int loop_num, int* progress)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (progress) {
    if (tid == 0) atomicExch_system(progress, 1);
    while (atomicAdd_system(progress + 32, 0) == 0) {}
    if (atomicAdd_system(progress + 48, 0) == 0) atomicExch_system(progress + 64, 1);
  }

  for (int i=0; i < loop_num; i++)
  {
    // Atomic addition
    atomicAdd_system(&atom_arr[0], 10);

    // Atomic exchange
    atomicExch_system(&atom_arr[1], tid);

    // Atomic maximum
    atomicMax_system(&atom_arr[2], tid);

    // Atomic minimum
    atomicMin_system(&atom_arr[3], tid);

    // Atomic increment (modulo 17+1)
    //atomicInc_system((unsigned int *)&atom_arr[4], 17);

    // Atomic decrement
    //atomicDec_system((unsigned int *)&atom_arr[5], 137);

    // Atomic compare-and-swap
    atomicCAS_system(&atom_arr[6], tid-1, tid);

    // Bitwise atomic instructions

    // Atomic AND
    system_bitwise(&atom_arr[7], 2*tid+7, 0);

    // Atomic OR
    system_bitwise(&atom_arr[8], static_cast<int>(1u << (static_cast<unsigned>(tid) & 31u)), 1);

    // Atomic XOR
    system_bitwise(&atom_arr[9], tid, 2);
  }
  if (progress) atomicAdd_system(progress + 16, 1);
}


extern "C" __global__ void atomicKernel(int* counters, int loops, int* progress) {
  atomicBody(counters, loops, progress);
}
extern "C" __global__ void atomicKernelUntraced(int* counters, int loops, int*) {
  atomicBody(counters, loops, nullptr);
}
