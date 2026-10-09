#include <hip/hip_runtime.h>

#include <cstddef>

template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) data_t {
  T data[VEC_SIZE];
};

template <typename T, int vec_size>
__device__ void threads_copy_impl(const T* in, T* out, const size_t n) {
  const int block_work_size = blockDim.x * vec_size;
  const size_t index = static_cast<size_t>(blockIdx.x) * block_work_size + threadIdx.x * vec_size;
  if (index >= n) {
    return;
  }

  const size_t remaining = n - index;
  if (remaining < static_cast<size_t>(vec_size)) {
    for (size_t i = index; i < n; i++) {
      out[i] = in[i];
    }
  } else {
    using vec_t = data_t<T, vec_size>;
    const auto* in_vec = reinterpret_cast<const vec_t*>(in + index);
    auto* out_vec = reinterpret_cast<vec_t*>(out + index);
    *out_vec = *in_vec;
  }
}

#define DEFINE_COPY_KERNEL(NAME, TYPE, VEC) \
  extern "C" __global__ void NAME(const TYPE* in, TYPE* out, const size_t n) { \
    threads_copy_impl<TYPE, VEC>(in, out, n); \
  }

DEFINE_COPY_KERNEL(threads_copy_int1, int, 1)
DEFINE_COPY_KERNEL(threads_copy_int2, int, 2)
DEFINE_COPY_KERNEL(threads_copy_int4, int, 4)
DEFINE_COPY_KERNEL(threads_copy_int8, int, 8)
DEFINE_COPY_KERNEL(threads_copy_int16, int, 16)

DEFINE_COPY_KERNEL(threads_copy_short1, short, 1)
DEFINE_COPY_KERNEL(threads_copy_short2, short, 2)
DEFINE_COPY_KERNEL(threads_copy_short4, short, 4)
DEFINE_COPY_KERNEL(threads_copy_short8, short, 8)
DEFINE_COPY_KERNEL(threads_copy_short16, short, 16)

DEFINE_COPY_KERNEL(threads_copy_char1, char, 1)
DEFINE_COPY_KERNEL(threads_copy_char2, char, 2)
DEFINE_COPY_KERNEL(threads_copy_char4, char, 4)
DEFINE_COPY_KERNEL(threads_copy_char8, char, 8)
DEFINE_COPY_KERNEL(threads_copy_char16, char, 16)
