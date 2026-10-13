#include <cuda_runtime.h>
#include "pnfa.h"
static_assert(sizeof(State)==40 && sizeof(u32)==4,"State/u32 ABI");
#include "pnfa.cu"
