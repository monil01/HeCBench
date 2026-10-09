#include <cstdint>

template <typename scalar_t>
__device__ void scatter_one(const scalar_t* __restrict__ src,
                            const int64_t* __restrict__ idx,
                            scalar_t* __restrict__ out,
                            int64_t num_elems)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < num_elems) {
    out[idx[i]] = src[i];
  }
}

extern "C" __global__ void scatter_i8(const int8_t* src, const int64_t* idx, int8_t* out, int64_t num_elems)
{
  scatter_one<int8_t>(src, idx, out, num_elems);
}

extern "C" __global__ void scatter_i16(const int16_t* src, const int64_t* idx, int16_t* out, int64_t num_elems)
{
  scatter_one<int16_t>(src, idx, out, num_elems);
}

extern "C" __global__ void scatter_i32(const int32_t* src, const int64_t* idx, int32_t* out, int64_t num_elems)
{
  scatter_one<int32_t>(src, idx, out, num_elems);
}

extern "C" __global__ void scatter_i64(const int64_t* src, const int64_t* idx, int64_t* out, int64_t num_elems)
{
  scatter_one<int64_t>(src, idx, out, num_elems);
}

extern "C" __global__ void scatter_f32(const float* src, const int64_t* idx, float* out, int64_t num_elems)
{
  scatter_one<float>(src, idx, out, num_elems);
}

extern "C" __global__ void scatter_f64(const double* src, const int64_t* idx, double* out, int64_t num_elems)
{
  scatter_one<double>(src, idx, out, num_elems);
}
