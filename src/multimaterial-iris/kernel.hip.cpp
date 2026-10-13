#include <hip/hip_runtime.h>
#include "kernel_body.h"
#define FUSED
#define ccc_loop1 ccc_loop1_F
#define ccc_loop1_2 ccc_loop1_2_F
#define ccc_loop2 ccc_loop2_F
#define ccc_loop2_2 ccc_loop2_2_F
#define ccc_loop3 ccc_loop3_F
#include "kernel_body.h"
#undef ccc_loop1
#undef ccc_loop1_2
#undef ccc_loop2
#undef ccc_loop2_2
#undef ccc_loop3
#undef FUSED
#define FUSED
#define LINKED
#define ccc_loop1 ccc_loop1_FL
#define ccc_loop1_2 ccc_loop1_2_FL
#define ccc_loop2 ccc_loop2_FL
#define ccc_loop2_2 ccc_loop2_2_FL
#define ccc_loop3 ccc_loop3_FL
#include "kernel_body.h"
#undef ccc_loop1
#undef ccc_loop1_2
#undef ccc_loop2
#undef ccc_loop2_2
#undef ccc_loop3
#undef FUSED
#undef LINKED
