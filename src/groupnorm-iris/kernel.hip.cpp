#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "common.h"
#define TPB 1024
template <int WarpSize>
__device__ __forceinline__
float cg_reduce (cooperative_groups::thread_block_tile<WarpSize,
                                                       cooperative_groups::thread_block> &warp,
                 float sum)
{
  for (int offset = WarpSize/2; offset > 0; offset /= 2) {
    sum += warp.shfl_xor(sum, offset);
  }
  return sum;
}

// Essentially taken from llm.c's kernel 5
// using kernel 5 because for images, each "channel" is effectively
// H * W * group_size, which is quite large
// One block per group of group_size channels: B * C / group_size (B * n_groups) blocks
extern "C" __global__ void groupnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B, int C, int img_size, int group_size, int n_groups
) {
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    __shared__ float shared_sum[TPB/WarpSize];
    __shared__ float shared_sum2[TPB/WarpSize];
    int num_warps = blockDim.x / WarpSize;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;

    // pixels in each group
    int group_pixels = img_size * group_size;

    // group index for each batch ranges from 0 to n_groups-1
    int g = blockIdx.x % n_groups;

    // move pointers
    x += blockIdx.x * group_pixels;
    out += blockIdx.x * group_pixels;
    // each block acceses group_size channels
    weight += g * group_size;
    bias += g * group_size;

    float thread_sum = 0.0f;
    float thread_sum2 = 0.0f;

    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        float val = x[i];
        thread_sum += val;
        thread_sum2 += val * val;
    }

    // reduce from all warps
    float warp_sum = cg_reduce<WarpSize>(warp, thread_sum);
    float warp_sum2 = cg_reduce<WarpSize>(warp, thread_sum2);
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();

    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = cg_reduce<WarpSize>(warp, warp_sum);
    float block_sum2 = cg_reduce<WarpSize>(warp, warp_sum2);

    block_sum /= group_pixels;
    block_sum2 /= group_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    if (mean != nullptr && threadIdx.x == 0) {
        mean[blockIdx.x] = m;
    }
    if (rstd != nullptr && threadIdx.x == 0) {
        rstd[blockIdx.x] = s;
    }

    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        int c = i / img_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c] + bias[c];
    }
}

extern "C" __global__ void groupnorm_forward_kernel2(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B, int C, int img_size, int group_size, int n_groups
) {
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    __shared__ float shared_sum[TPB/WarpSize];
    __shared__ float shared_sum2[TPB/WarpSize];
    int num_warps = blockDim.x / WarpSize;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;

    // pixels in each group
    int group_pixels = img_size * group_size;

    // group index for each batch ranges from 0 to n_groups-1
    int g = blockIdx.x % n_groups;

    // move pointers
    x += blockIdx.x * group_pixels;
    out += blockIdx.x * group_pixels;
    // each block acceses group_size channels
    weight += g * group_size;
    bias += g * group_size;

    // vectorize the accesses
    int vec_count = group_pixels / 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float4 thread_sum_v4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 thread_sum2_v4 = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      float4 v = x_vec[i];
      thread_sum_v4.x  += v.x;
      thread_sum_v4.y  += v.y;
      thread_sum_v4.z  += v.z;
      thread_sum_v4.w  += v.w;
      thread_sum2_v4.x += v.x * v.x;
      thread_sum2_v4.y += v.y * v.y;
      thread_sum2_v4.z += v.z * v.z;
      thread_sum2_v4.w += v.w * v.w;
    }

    float thread_sum = thread_sum_v4.x + thread_sum_v4.y + thread_sum_v4.z + thread_sum_v4.w;
    float thread_sum2 = thread_sum2_v4.x + thread_sum2_v4.y + thread_sum2_v4.z + thread_sum2_v4.w;

    for (int i = vec_count * 4 + threadIdx.x; i < group_pixels; i += blockDim.x) {
        float val = x[i];
        thread_sum  += val;
        thread_sum2 += val * val;
    }

    // reduce from all warps
    float warp_sum = cg_reduce<WarpSize>(warp, thread_sum);
    float warp_sum2 = cg_reduce<WarpSize>(warp, thread_sum2);
    // store warp sum into shared memory
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();

    // load warp sums from shared memory
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float block_sum = cg_reduce<WarpSize>(warp, warp_sum);
    float block_sum2 = cg_reduce<WarpSize>(warp, warp_sum2);

    block_sum /= group_pixels;
    block_sum2 /= group_pixels;
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);
    if (mean != nullptr && threadIdx.x == 0) {
        mean[blockIdx.x] = m;
    }
    if (rstd != nullptr && threadIdx.x == 0) {
        rstd[blockIdx.x] = s;
    }

    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        int c = i / img_size;
        float n = s * (x[i] - m);
        out[i] = n * weight[c] + bias[c];
    }
}

// most similar to kernel 2 in llm.c
// not doing any float16 optimizations yet
// main change is to have each block allocated to a single group
extern "C" __global__ void groupnorm_backward_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ weight,
    float* __restrict__ dx,
    float* __restrict__ dweight,
    float* __restrict__ dbias,
    int B, int C, int img_size, int group_size, int n_groups
) {
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    __shared__ float shared_sum[TPB/WarpSize];
    __shared__ float shared_sum2[TPB/WarpSize];
    int num_warps = blockDim.x / WarpSize;
    int warp_id = threadIdx.x / WarpSize;
    int lane_id = threadIdx.x % WarpSize;
    int group_pixels = img_size * group_size;
    // group index
    int g = blockIdx.x % n_groups;

    // move pointers
    dout += blockIdx.x * img_size * group_size;
    x += blockIdx.x * img_size * group_size;
    dx += blockIdx.x * img_size * group_size;
    weight += g * group_size;
    dweight += g * group_size;
    dbias += g * group_size;

    float m_val = mean[blockIdx.x];
    float rstd_val = rstd[blockIdx.x];

    // calculate the two mean terms in the group dimension
    // first is dout * weight, and second is dout * weight * norm
    // where norm = (x - mean) * rstd
    float w_dout_thread = 0.0f;
    float w_dout_norm_thread = 0.0f;
    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        int c = i / img_size;
        float cur_w_dout = weight[c] * dout[i];
        w_dout_thread += cur_w_dout;
        float norm = (x[i] - m_val) * rstd_val;
        w_dout_norm_thread += cur_w_dout * norm;
    }
    // warp reduce
    float w_dout_warp = cg_reduce<WarpSize>(warp, w_dout_thread);
    float w_dout_norm_warp = cg_reduce<WarpSize>(warp, w_dout_norm_thread);
    // store warp sum in shared mem
    shared_sum[warp_id] = w_dout_warp;
    shared_sum2[warp_id] = w_dout_norm_warp;
    __syncthreads();

    // load warp sums from shared memory
    w_dout_warp = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    w_dout_norm_warp = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    float w_dout_block = cg_reduce<WarpSize>(warp, w_dout_warp);
    float w_dout_norm_block = cg_reduce<WarpSize>(warp, w_dout_norm_warp);
    w_dout_block /= group_pixels;
    w_dout_norm_block /= group_pixels;

    // update dx
    for (int i = threadIdx.x; i < group_pixels; i += blockDim.x) {
        // in bounds of image
        // accumulate dw and db
        float dout_val = dout[i];
        float norm = (x[i] - m_val) * rstd_val;

        // update dx
        int c = (i / img_size) % group_size;
        float w_dout = weight[c] * dout_val;
        dx[i] = (w_dout - w_dout_block - norm * w_dout_norm_block) * rstd_val;
    }
    // update dw and db
    // use different methods when the image size is large or small

    // if the image size is larger than the block size
    // loop over the channels and use the whole block on each channel
    // otherwise, assign each warp to a channel
    // in either case image size must be larger than the warp size
    //assert(img_size % warp.size() == 0);
    //assert(blockDim.x % warp.size() == 0);
    if (img_size % blockDim.x == 0) {
        for (int c = 0; c < group_size; c++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;
            for (int i = threadIdx.x; i < img_size; i += blockDim.x) {
                float dout_val = dout[i];
                db_thread += dout_val;
                float norm = (x[i] - m_val) * rstd_val;
                dw_thread += dout_val * norm;
            }

            // move pointers
            dout += img_size;
            x += img_size;

            // warp reduce
            float dw_warp = cg_reduce<WarpSize>(warp, dw_thread);
            float db_warp = cg_reduce<WarpSize>(warp, db_thread);
            ////// store warp sum in shared mem
            if (lane_id == 0) {
                shared_sum[warp_id] = dw_warp;
                shared_sum2[warp_id] = db_warp;
            }
            __syncthreads();
            // use the first thread to reduce the shared memory sums and save to global memory
            if (threadIdx.x == 0) {
                float dw_block = 0.0f;
                float db_block = 0.0f;
                for (int i = 0; i < num_warps; i++) {
                    dw_block += shared_sum[i];
                    db_block += shared_sum2[i];
                }
                atomicAdd(dweight + c, dw_block);
                atomicAdd(dbias + c, db_block);
            }
        }
    } else {
        // if group size is large, need to loop over the group channels with the whole block
        int block_reps = ceil_div(group_size, num_warps);
        for (int br = 0; br < block_reps; br++) {
            float dw_thread = 0.0f;
            float db_thread = 0.0f;

            int ch = br * num_warps + warp_id;
            if (ch < group_size) {
                const float* dout_ch = dout + ch * img_size;
                const float* x_ch = x + ch * img_size;
                for (int i = lane_id; i < img_size; i += warp.size()) {
                    float dout_val = dout_ch[i];
                    db_thread += dout_val;
                    float norm = (x_ch[i] - m_val) * rstd_val;
                    dw_thread += dout_val * norm;
                }

                // warp reduce
                float dw_warp = cg_reduce<WarpSize>(warp, dw_thread);
                float db_warp = cg_reduce<WarpSize>(warp, db_thread);
                // since each warp takes care of an entire image
                // directly store result
                if (lane_id == 0) {
                    atomicAdd(dweight + ch, dw_warp);
                    atomicAdd(dbias + ch, db_warp);
                }
            }
        }
    }
}
