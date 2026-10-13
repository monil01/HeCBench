#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub=hipcub;
#define CEILDIV(x,y) (((x)+(y)-1)/(y))
template <typename scalar_t>
__device__ void _moe_align_block_size(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts,
    int padded_num_experts, int experts_per_warp, int block_size,
    int numel, int* __restrict__ cumsum, int max_num_tokens_padded,
    int max_num_m_blocks, int model_offset, int inactive_expert_id,
    int topk_num, int* token_mask)
{
  __shared__ int shared_counts[1024];

  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;
  int cumsum_offset = (num_experts + 1) * model_offset;

  // Use separate threadblocks to fill sorted_token_ids.
  // This is safe since the current kernel does not use sorted_token_ids.
  if (blockIdx.x % 2) {
    // Initialize sorted_token_ids with numel
    for (int it = threadIdx.x; it < max_num_tokens_padded;
         it += blockDim.x) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / warpSize;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
              mask);
  }

  __syncthreads();

  // Compute prefix sum over token counts per expert
  using BlockScan = cub::BlockScan<int, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[cumsum_offset + expert_id] = cumsum_val;
  }

  if (expert_id == num_experts) {
    total_tokens_post_pad[model_offset] = cumsum_val;
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[cumsum_offset + threadIdx.x];
         i < cumsum[cumsum_offset + threadIdx.x + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining expert_ids with 0
  const int fill_start_idx =
      cumsum[cumsum_offset + num_experts] / block_size + threadIdx.x;
  for (int i = fill_start_idx; i < max_num_m_blocks; i += blockDim.x) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }
}

template <typename scalar_t, int fill_threads>
__device__ void _moe_align_block_size_small_batch_expert(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids,
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ expert_map, int num_experts, int block_size,
    int numel, int max_num_tokens_padded, int max_num_m_blocks,
    int inactive_expert_id, int model_offset, int topk_num,
    int* token_mask)
{
  // Compute input buffer offsets. Typically these will all be 0, except when
  // using Multi LoRA.
  int sorted_token_ids_offset = max_num_tokens_padded * model_offset;
  int expert_ids_offset = max_num_m_blocks * model_offset;

  // Use an additional group of threads to fill sorted_token_ids.
  // Since the current kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    for (int it = threadIdx.x; it < max_num_tokens_padded;
         it += fill_threads) {
      sorted_token_ids[sorted_token_ids_offset + it] = numel;
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    //__syncthreads();
    //__syncthreads();
    return;
  }

  const int tid = threadIdx.x - fill_threads;
  const int stride = blockDim.x - fill_threads;

  __shared__ int shared_mem[4225];
  int* cumsum = shared_mem;
  int* tokens_cnts = (int*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int mask = token_mask == nullptr ? 1 : token_mask[i / topk_num];
    tokens_cnts[(tid + 1) * num_experts + expert_id] += mask;
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
    }
    total_tokens_post_pad[model_offset] =
        static_cast<int>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[expert_ids_offset + i / block_size] = tid;
    }
  }

  // Fill remaining expert_ids with 0
  const int fill_start_idx = cumsum[num_experts] / block_size + tid;
  for (int i = fill_start_idx; i < max_num_m_blocks; i += stride) {
    expert_ids[expert_ids_offset + i] = inactive_expert_id;
  }

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid expert
      if (expert_id == -1) continue;
    }
    int rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      sorted_token_ids[sorted_token_ids_offset + rank_post_pad] = i;
      ++tokens_cnts[tid * num_experts + expert_id];
    }
  }
}

template <typename scalar_t>
__device__ void _count_and_sort_expert_tokens(
    const scalar_t* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids, int* __restrict__ cumsum_buffer,
    int* __restrict__ expert_map, int numel, int num_experts,
    int max_num_tokens_padded, int* __restrict__ token_mask,
    int model_offset, int topk_num)
{
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.y;

  for (int i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }

    if (expert_map != nullptr) {
      expert_id = expert_map[expert_id];
      // filter invalid experts
      if (expert_id == -1) continue;
    }

    if (token_mask == nullptr || token_mask[i / topk_num]) {
      int rank_post_pad = atomicAdd(&cumsum_buffer[(model_offset * (num_experts + 1)) + expert_id], 1);
      sorted_token_ids[max_num_tokens_padded * model_offset + rank_post_pad] = i;
    }
  }
}

extern "C" __global__ void align_large(const int* topk,int* sorted,int* experts,int* total,
  int* map,int n,int padded,int perwarp,int block,int numel,int* cumsum,int maxpad,int topk_num) {
  _moe_align_block_size(topk,sorted,experts,total,map,n,padded,perwarp,block,numel,cumsum,maxpad,
    CEILDIV(maxpad,block),0,0,topk_num,nullptr);
}
extern "C" __global__ void align_small(const int* topk,int* sorted,int* experts,int* total,
  int* map,int n,int block,int numel,int maxpad,int topk_num) {
  _moe_align_block_size_small_batch_expert<int,256>(topk,sorted,experts,total,map,n,block,numel,maxpad,
    CEILDIV(maxpad,block),0,0,topk_num,nullptr);
}
extern "C" __global__ void sort_tokens(const int* topk,int* sorted,int* cumsum,int* map,
  int numel,int n,int maxpad,int topk_num) {
  _count_and_sort_expert_tokens(topk,sorted,cumsum,map,numel,n,maxpad,nullptr,0,topk_num);
}
