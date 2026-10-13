#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#define PI 3.14159265359f
#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_ADD 256
#include "kernels.hpp"
