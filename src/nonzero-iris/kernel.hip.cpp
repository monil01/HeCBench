#include <hip/hip_runtime.h>
#include <stdint.h>

#define MAX_DIMS 2

template <typename T>
__device__ inline bool is_nonzero(T value)
{
  return value != static_cast<T>(0);
}

template <typename T>
__device__ void count_nonzero_impl(const T* in, uint64_t* nzeros, int64_t num_items)
{
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < num_items && is_nonzero(in[index])) {
    atomicAdd(nzeros, static_cast<uint64_t>(1));
  }
}

template <typename T>
__device__ void compact_nonzero_impl(const T* in, int64_t* out, const uint64_t* offsets, int64_t num_items)
{
  if (threadIdx.x != 0) return;
  uint64_t pos = offsets[blockIdx.x];
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * blockDim.x;
  for (int64_t index = begin; index < begin + blockDim.x && index < num_items; ++index) {
    if (is_nonzero(in[index])) out[pos++] = index;
  }
}

// Ordered block offsets replace the host hipCUB DeviceSelect operation.
template <typename T>
__device__ void offsets_impl(const T* in, uint64_t* offsets, int64_t n) {
  if (blockIdx.x || threadIdx.x) return;
  uint64_t count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (i % 256 == 0) offsets[i / 256] = count;
    count += is_nonzero(in[i]);
  }
}
#define OFFSET_WRAPPER(suffix, type) \
extern "C" __global__ void offsets_##suffix(const type* in, uint64_t* offsets, int64_t n) { offsets_impl(in, offsets, n); }
OFFSET_WRAPPER(i8, int8_t)
OFFSET_WRAPPER(i16, int16_t)
OFFSET_WRAPPER(i32, int32_t)
OFFSET_WRAPPER(f32, float)
OFFSET_WRAPPER(f64, double)

extern "C" __global__ void count_nonzero_i8(const int8_t* in, uint64_t* nzeros, int64_t num_items)
{
  count_nonzero_impl(in, nzeros, num_items);
}

extern "C" __global__ void count_nonzero_i16(const int16_t* in, uint64_t* nzeros, int64_t num_items)
{
  count_nonzero_impl(in, nzeros, num_items);
}

extern "C" __global__ void count_nonzero_i32(const int32_t* in, uint64_t* nzeros, int64_t num_items)
{
  count_nonzero_impl(in, nzeros, num_items);
}

extern "C" __global__ void count_nonzero_f32(const float* in, uint64_t* nzeros, int64_t num_items)
{
  count_nonzero_impl(in, nzeros, num_items);
}

extern "C" __global__ void count_nonzero_f64(const double* in, uint64_t* nzeros, int64_t num_items)
{
  count_nonzero_impl(in, nzeros, num_items);
}

extern "C" __global__ void compact_nonzero_i8(const int8_t* in, int64_t* out, uint64_t* nzeros, int64_t num_items)
{
  compact_nonzero_impl(in, out, nzeros, num_items);
}

extern "C" __global__ void compact_nonzero_i16(const int16_t* in, int64_t* out, uint64_t* nzeros, int64_t num_items)
{
  compact_nonzero_impl(in, out, nzeros, num_items);
}

extern "C" __global__ void compact_nonzero_i32(const int32_t* in, int64_t* out, uint64_t* nzeros, int64_t num_items)
{
  compact_nonzero_impl(in, out, nzeros, num_items);
}

extern "C" __global__ void compact_nonzero_f32(const float* in, int64_t* out, uint64_t* nzeros, int64_t num_items)
{
  compact_nonzero_impl(in, out, nzeros, num_items);
}

extern "C" __global__ void compact_nonzero_f64(const double* in, int64_t* out, uint64_t* nzeros, int64_t num_items)
{
  compact_nonzero_impl(in, out, nzeros, num_items);
}

extern "C" __global__ void write_indices(int64_t* inp,
                                         int64_t ncols,
                                         int ndim,
                                         int64_t nzero,
                                         int64_t num_items)
{
  const int64_t index = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.x) * blockDim.x;
  if (index < nzero && index < num_items) {
    int64_t div = 1;
    const int64_t idx_flat = inp[index];
    for (int dim = MAX_DIMS; dim >= 0; dim--) {
      if (dim > ndim - 1) continue;
      const int64_t dim_size = dim == 1 ? ncols : num_items / ncols;
      inp[index + dim * nzero] = (idx_flat / div) % dim_size;
      div *= dim_size;
    }
  }
}
