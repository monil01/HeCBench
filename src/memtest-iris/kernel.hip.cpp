#include <hip/hip_runtime.h>

extern "C" {
#include "../memtest-cuda/kernels.h"
}

extern "C" __global__ void kernel_reset_error(unsigned* err_count)
{
  *err_count = 0;
}

extern "C" __global__ void kernel_flush_error(const unsigned* err_count)
{
  (void)err_count;
}
