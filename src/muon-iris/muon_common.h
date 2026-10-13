#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <chrono>
#include <random>
#include <algorithm>
#include "reference.h"

// The Muon optimizer is applied to a single 2D hidden weight matrix of shape
// (rows x cols) == (out_features x in_features) of a Linear layer. The momentum
// buffer, the gradient and the weights are kept in float32; the Newton-Schulz
// orthogonalization runs in bfloat16 (with float32 accumulation), as in
// the reference implementation https://github.com/KellerJordan/Muon .
//
// This version uses hipBLAS bf16 GEMMs with fp32 accumulation for
// the Newton-Schulz matmuls, and fuses the elementwise steps:
//   * momentum update + fp32->bf16 conversion  -> momentum_update
//   * Newton-Schulz spectral scaling            -> folded into param_update

typedef __hip_bfloat16 bf16;

#define CHECK_HIP(call)                                                        \
  do {                                                                         \
    int err__ = (call);                                                 \
    if (err__ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,             \
              "IRIS status failure");                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// 8-byte (4 x bf16) vector for 128-bit access
struct __align__(8) bf16x4 { __hip_bfloat162 lo, hi; };


#define T_TILE 32
#define T_BROWS 8
enum GemmOp { GEMM_N, GEMM_T };
