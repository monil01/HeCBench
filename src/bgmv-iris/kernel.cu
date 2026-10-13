#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// ============================================================
// BGMV-Shrink kernel
//
//   output[token, r] += scaling * sum_k( input[token, k] * W[lora_id, r, k] )
//   where lora_id = lora_indices[token]
//
//   input  : [num_tokens, hidden_size]   (fp16)
//   weight : [num_loras,  lora_rank, hidden_size]  (fp16, col-major => row-major here)
//   output : [num_tokens, lora_rank]     (fp32, accumulated)
//   lora_indices : [num_tokens]          (int32)  -- per-token LoRA index
// ============================================================
template<int BLOCK_SIZE = 128>
__device__ void bgmv_shrink_kernel(
    float*        output,          // [num_tokens, lora_rank]
    const __half* input,           // [num_tokens, hidden_size]
    const __half* weights,         // [num_loras, lora_rank, hidden_size]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    float         scaling)
{
    // grid : (num_tokens, lora_rank)
    int token_idx = blockIdx.x;
    int rank_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || rank_idx >= lora_rank) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const __half* inp = input   + token_idx * hidden_size;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const __half* wt  = weights + lora_id * lora_rank * hidden_size
                                + rank_idx * hidden_size;

    // each thread processes partial dot product along hidden dimension
    float acc = 0.0f;
    for (int k = tid; k < hidden_size; k += BLOCK_SIZE) {
        acc += __half2float(inp[k]) * __half2float(wt[k]);
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    acc = BlockReduce(temp_storage1).Sum(acc);
    if (tid == 0)
      output[token_idx * lora_rank + rank_idx] += scaling * acc;
}

__device__
inline void unpack(const half* __restrict__ in,
                         half* __restrict__ out)
{
  auto v4 = *reinterpret_cast<const float4*>(in);
  auto h2 = reinterpret_cast<const __half2*>(&v4);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
      out[2*i]     = __low2half(h2[i]);
      out[2*i + 1] = __high2half(h2[i]);
  }
}

template<int BLOCK_SIZE>
__device__ void bgmv_shrink_kernel_vec(
    float*        output,          // [num_tokens, lora_rank]
    const __half* input,           // [num_tokens, hidden_size]
    const __half* weights,         // [num_loras, lora_rank, hidden_size]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    float         scaling)
{
    // grid : (num_tokens, lora_rank)
    int token_idx = blockIdx.x;
    int rank_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    if (token_idx >= num_tokens || rank_idx >= lora_rank) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, hidden_size] -> inputs[batch_id, :]
    const __half* inp = input   + token_idx * hidden_size;

    // weights: [num_loras, rank, hidden_size] -> weights[lora_idx, rank_id, :]
    const __half* wt  = weights + lora_id * lora_rank * hidden_size
                                + rank_idx * hidden_size;

    int offset = tid * 8;

    float acc = 0.f;

    while (offset < hidden_size) {
      const auto remain = hidden_size - offset;
      if (remain >= 8) {
        __half inp_vec[8], wt_vec[8];
        unpack(inp + offset, inp_vec);
        unpack(wt + offset, wt_vec);

        for (int i = 0; i < 8; i++)
          acc += __half2float(inp_vec[i]) * __half2float(wt_vec[i]);

      } else {
        // Partial vector processing: handle remaining elements less than vec_size
        for (int k = 0; k < remain; k++) {
          acc += __half2float(inp[offset + k]) * __half2float(wt[offset + k]);
        }
      }
      offset += BLOCK_SIZE * 8;  // next position handled by current thread
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    acc = BlockReduce(temp_storage1).Sum(acc);
    if (tid == 0)
      output[token_idx * lora_rank + rank_idx] += scaling * acc;
}

// ============================================================
// BGMV-Expand kernel
//
//   output[token, n] = sum_r(input[token, r] * W[lora_id, n, r])
//                      + (add_to_output ? outputs[token, n] : 0)
//   where lora_id = lora_indices[token]
//
//   input  : [num_tokens, lora_rank]     (fp32)
//   weight : [num_loras,  hidden_size, lora_rank]  (fp16)
//   output : [num_tokens, hidden_size]   (fp16)
//   add_to_output
// ============================================================
template<int BLOCK_SIZE>
__device__ void bgmv_expand_kernel(
    __half*       output,          // [num_tokens, hidden_size]
    const float*  input,           // [num_tokens, lora_rank]
    const __half* weights,         // [num_loras, hidden_size, lora_rank]
    const int*    lora_indices,    // [num_tokens]
    int           num_tokens,
    int           hidden_size,
    int           lora_rank,
    bool          add_to_output)
{
    // grid : (num_tokens, hidden_size)
    int token_idx  = blockIdx.x;
    int hidden_idx = blockIdx.y;
    int tid        = threadIdx.x;

    if (token_idx >= num_tokens || hidden_idx >= hidden_size) return;

    int lora_id = lora_indices[token_idx];

    // inputs: [batch_size, lora_rank] -> inputs[batch_id, :]
    const float*  inp = input   + token_idx  * lora_rank;

    // weights: [num_loras, hidden_size, rank] -> weights[lora_idx, rank_id, :]
    const __half* wt  = weights + lora_id    * hidden_size * lora_rank
                                + hidden_idx * lora_rank;

    // each thread processes partial dot product along rank dimension
    float acc = 0.0f;
    for (int r = tid; r < lora_rank; r += BLOCK_SIZE) {
        acc += inp[r] * __half2float(wt[r]);
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    acc = BlockReduce(temp_storage1).Sum(acc);
    if (tid == 0) {
      int out_idx = token_idx * hidden_size + hidden_idx;
      float prev  = add_to_output ? __half2float(output[out_idx]) : 0.f;
      output[out_idx] = __float2half(prev + acc);
    }
}


extern "C" __global__ void bgmv_shrink_32(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel<32>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_shrink_vec_32(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel_vec<32>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_expand_32(__half* out, const float* in, const __half* wt, const int* ids, int t, int h, int r, bool add) {
    bgmv_expand_kernel<32>(out, in, wt, ids, t, h, r, add);
}

extern "C" __global__ void bgmv_shrink_64(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel<64>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_shrink_vec_64(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel_vec<64>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_expand_64(__half* out, const float* in, const __half* wt, const int* ids, int t, int h, int r, bool add) {
    bgmv_expand_kernel<64>(out, in, wt, ids, t, h, r, add);
}

extern "C" __global__ void bgmv_shrink_128(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel<128>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_shrink_vec_128(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel_vec<128>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_expand_128(__half* out, const float* in, const __half* wt, const int* ids, int t, int h, int r, bool add) {
    bgmv_expand_kernel<128>(out, in, wt, ids, t, h, r, add);
}

extern "C" __global__ void bgmv_shrink_256(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel<256>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_shrink_vec_256(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel_vec<256>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_expand_256(__half* out, const float* in, const __half* wt, const int* ids, int t, int h, int r, bool add) {
    bgmv_expand_kernel<256>(out, in, wt, ids, t, h, r, add);
}

extern "C" __global__ void bgmv_shrink_512(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel<512>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_shrink_vec_512(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel_vec<512>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_expand_512(__half* out, const float* in, const __half* wt, const int* ids, int t, int h, int r, bool add) {
    bgmv_expand_kernel<512>(out, in, wt, ids, t, h, r, add);
}

extern "C" __global__ void bgmv_shrink_1024(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel<1024>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_shrink_vec_1024(float* out, const __half* in, const __half* wt, const int* ids, int t, int h, int r, float scaling) {
    bgmv_shrink_kernel_vec<1024>(out, in, wt, ids, t, h, r, scaling);
}

extern "C" __global__ void bgmv_expand_1024(__half* out, const float* in, const __half* wt, const int* ids, int t, int h, int r, bool add) {
    bgmv_expand_kernel<1024>(out, in, wt, ids, t, h, r, add);
}
