#include <hip/hip_runtime.h>
#include "helper/rngpu.hpp"
#include "helper/cuda_helpers.cuh"
#include "config.h"
#include "updates_and_measures.cuh"
#include "bit_vector_kernels.cuh"
extern "C" __global__ void iris_initFactor(uint32_t * Ab, const uint32_t height, const uint8_t factorDim, const uint32_t seed, const float threshold){initFactor(Ab,height,factorDim,seed,threshold);}
extern "C" __global__ void iris_computeDistanceRows(const uint32_t * Ab, const uint32_t * Bb, const uint32_t * Cb, const uint32_t height, const uint32_t width, const uint32_t padded_width, const uint8_t factorDim, const int weight, float *global_error){computeDistanceRows(Ab,Bb,Cb,height,width,padded_width,factorDim,weight,global_error);}
extern "C" __global__ void iris_computeDistanceRowsShared(const uint32_t * Ab, const uint32_t * Bb, const uint32_t * Cb, const uint32_t height, const uint32_t width, const uint32_t padded_width, const uint8_t factorDim, const float weight, float *global_error){computeDistanceRowsShared(Ab,Bb,Cb,height,width,padded_width,factorDim,weight,global_error);}
extern "C" __global__ void iris_vectorMatrixMultCompareRowWarpShared(uint32_t * A, const uint32_t * B, const uint32_t * C, const uint32_t height, const uint32_t width, const uint32_t padded_width, const uint8_t factorDim, const uint32_t startrow, float *global_error, const uint32_t seed, const float temperature, const float flipManyChance, const uint32_t flipManyDepth, const float weight){vectorMatrixMultCompareRowWarpShared(A,B,C,height,width,padded_width,factorDim,startrow,global_error,seed,temperature,flipManyChance,flipManyDepth,weight);}
extern "C" __global__ void iris_vectorMatrixMultCompareColWarpShared(const uint32_t * A, uint32_t * B, const uint32_t * C, const uint32_t height, const uint32_t width, const uint32_t padded_width, const uint8_t factorDim, const uint32_t startcol, float *global_error, const uint32_t seed, const float temperature, const float flipManyChance, const uint32_t flipManyDepth, const float weight){vectorMatrixMultCompareColWarpShared(A,B,C,height,width,padded_width,factorDim,startcol,global_error,seed,temperature,flipManyChance,flipManyDepth,weight);}
