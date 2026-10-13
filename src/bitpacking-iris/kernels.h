#include <stdint.h>
#include <stddef.h>

// Match the source's 256-thread, at-most-4096-block strided reduction.
extern "C" __global__ void scan(int32_t* lows, int32_t* highs,
                               const int32_t* input, size_t n) {
  __shared__ int32_t lo[256], hi[256];
  int32_t a = INT32_MAX, b = INT32_MIN;
  for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n;
       i += gridDim.x * 256) {
    a = min(a, input[i]); b = max(b, input[i]);
  }
  lo[threadIdx.x] = a; hi[threadIdx.x] = b;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) {
      lo[threadIdx.x] = min(lo[threadIdx.x], lo[threadIdx.x+d]);
      hi[threadIdx.x] = max(hi[threadIdx.x], hi[threadIdx.x+d]);
    }
    __syncthreads();
  }
  if (!threadIdx.x) { lows[blockIdx.x] = lo[0]; highs[blockIdx.x] = hi[0]; }
}

extern "C" __global__ void config(const int32_t* lows, const int32_t* highs,
                                 unsigned char* bits, int32_t* minimum,
                                 size_t blocks) {
  __shared__ int32_t lo[256], hi[256];
  int32_t a = INT32_MAX, b = INT32_MIN;
  for (size_t i = threadIdx.x; i < blocks; i += 256) {
    a = min(a, lows[i]); b = max(b, highs[i]);
  }
  lo[threadIdx.x] = a; hi[threadIdx.x] = b;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) {
      lo[threadIdx.x] = min(lo[threadIdx.x], lo[threadIdx.x+d]);
      hi[threadIdx.x] = max(hi[threadIdx.x], hi[threadIdx.x+d]);
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    *minimum = lo[0];
    uint32_t range = uint32_t(hi[0]) - uint32_t(lo[0]);
    *bits = range ? 32 - __clz(range) : 0;
  }
}

// Each thread owns a 32-bit output word, preserving the source's little-endian
// bit stream without atomics or overlapping writes.
extern "C" __global__ void pack(const unsigned char* width, const int32_t* minimum,
                               uint32_t* output, const int32_t* input,
                               size_t n, size_t words) {
  unsigned bits = *width;
  for (size_t word = blockIdx.x * blockDim.x + threadIdx.x; word < words;
       word += gridDim.x * blockDim.x) {
    uint32_t value = 0;
    if (bits) {
      size_t start = word * 32;
      size_t first = start / bits;
      size_t end = min(n, (start + 32 + bits - 1) / bits);
      for (size_t i = first; i < end; ++i) {
        uint32_t delta = uint32_t(input[i]) - uint32_t(*minimum);
        int shift = int(i * bits - start);
        value |= shift >= 0 ? delta << shift : delta >> -shift;
      }
    }
    output[word] = value;
  }
}
