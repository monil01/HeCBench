#define NOMINMAX

#include <algorithm> //min  max
#include <vector>
#include <math.h>
#include <hip/hip_runtime.h>
#include "dmem.hpp"

#include "indices.hpp"
#include "params.hpp"

//2DDCT - has to be consistent with dct8x8.cu
#define KER2_BLOCK_WIDTH          128

//Exception handling
#include <sstream>
//Debug
#include <fstream>
#include <iostream>

//Extern kernels

extern "C" void run_block_matching(
  BMBuffer &image,
  BMBuffer &stacks,
  BMBuffer &num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const uint2 start_point,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size
);

extern "C" void run_get_block(
  const uint2 start_point,
  BMBuffer &image,
  BMBuffer &stacks,
  BMBuffer &num_patches_in_stack,
  BMBuffer &patch_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_DCT2D8x8(
  BMBuffer &d_transformed_stacks,
  BMBuffer &d_gathered_stacks,
  const uint size,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_hard_treshold_block(
  const uint2 start_point,
  BMBuffer &patch_stack,
  BMBuffer &w_P,
  BMBuffer &num_patches_in_stack,
  const uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size
);

extern "C" void run_IDCT2D8x8(
  BMBuffer &d_gathered_stacks,
  BMBuffer &d_transformed_stacks,
  const uint size,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_aggregate_block(
  const uint2 start_point,
  BMBuffer &patch_stack,
  BMBuffer &w_P,
  BMBuffer &stacks,
  BMBuffer &kaiser_window,
  BMBuffer &numerator,
  BMBuffer &denominator,
  BMBuffer &num_patches_in_stack,
  const uint2 image_dim,
  const uint2 stacks_dim,
  const Params params,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_aggregate_final(
  BMBuffer &numerator,
  BMBuffer &denominator,
  const uint2 image_dim,
  BMBuffer &denoised_noisy_image,
  const dim3 num_threads,
  const dim3 num_blocks
);

extern "C" void run_wiener_filtering(
  const uint2 start_point,
  BMBuffer &patch_stack,
  BMBuffer &patch_stack_basic,
  BMBuffer &w_P,
  BMBuffer &num_patches_in_stack,
  uint2 stacks_dim,
  const Params params,
  const uint sigma,
  const dim3 num_threads,
  const dim3 num_blocks,
  const uint shared_memory_size
);

