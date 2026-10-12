#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float sigmoid_device(float in)
{
  return 1.0f / (1.0f + expf(-in));
}

extern "C" __global__ void gru_cell_forward(const half* __restrict__ input,
                                             const half* __restrict__ hidden,
                                             const half* __restrict__ bias1,
                                             const half* __restrict__ bias2,
                                             const half* __restrict__ hx,
                                             half* __restrict__ hy,
                                             half* __restrict__ storage,
                                             int hsz,
                                             int total_elements)
{
  const int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= total_elements) return;

  const int batch_idx = linear_idx / hsz;
  const int hidden_idx = linear_idx % hsz;
  const int input_offset = batch_idx * 3 * hsz + hidden_idx;

  const half ir = input[input_offset + 0 * hsz];
  const half ii = input[input_offset + 1 * hsz];
  const half in = input[input_offset + 2 * hsz];
  const half hr = hidden[input_offset + 0 * hsz];
  const half hi = hidden[input_offset + 1 * hsz];
  const half hn = hidden[input_offset + 2 * hsz];
  const half hxv = hx[linear_idx];

  const half b1r = bias1[hidden_idx + 0 * hsz];
  const half b1i = bias1[hidden_idx + 1 * hsz];
  const half b1n = bias1[hidden_idx + 2 * hsz];
  const half b2r = bias2[hidden_idx + 0 * hsz];
  const half b2i = bias2[hidden_idx + 1 * hsz];
  const half b2n = bias2[hidden_idx + 2 * hsz];

  const float rg = sigmoid_device(static_cast<float>(ir) + static_cast<float>(hr) +
                                  static_cast<float>(b1r) + static_cast<float>(b2r));
  const float ig = sigmoid_device(static_cast<float>(ii) + static_cast<float>(hi) +
                                  static_cast<float>(b1i) + static_cast<float>(b2i));
  float ng = static_cast<float>(in) + static_cast<float>(b1n) +
             rg * (static_cast<float>(hn) + static_cast<float>(b2n));
  ng = tanhf(ng);

  hy[linear_idx] = static_cast<half>(ng + ig * (static_cast<float>(hxv) - ng));

  const int store_offset = batch_idx * 5 * hsz + hidden_idx;
  storage[store_offset + 0 * hsz] = static_cast<half>(rg);
  storage[store_offset + 1 * hsz] = static_cast<half>(ig);
  storage[store_offset + 2 * hsz] = static_cast<half>(ng);
  storage[store_offset + 3 * hsz] = hxv;
  storage[store_offset + 4 * hsz] =
      static_cast<half>(static_cast<float>(hn) + static_cast<float>(b2n));
}
