#include <cstdint>

template <typename T>
__device__ void reverse2d_kernel(T* out,
                                 const T* in,
                                 int nrows,
                                 int ncols,
                                 int row_major,
                                 int along_rows,
                                 int total)
{
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;

  int row;
  int col;
  if (row_major) {
    row = idx / ncols;
    col = idx % ncols;
  } else {
    row = idx % nrows;
    col = idx / nrows;
  }

  const int dst_row = along_rows ? (nrows - row - 1) : row;
  const int dst_col = along_rows ? col : (ncols - col - 1);
  const int dst_idx = row_major ? (dst_row * ncols + dst_col) : (dst_col * nrows + dst_row);
  out[dst_idx] = in[idx];
}

extern "C" __global__ void reverse2d_u8(unsigned char* out,
                                         const unsigned char* in,
                                         int nrows,
                                         int ncols,
                                         int row_major,
                                         int along_rows,
                                         int total)
{
  reverse2d_kernel(out, in, nrows, ncols, row_major, along_rows, total);
}

extern "C" __global__ void reverse2d_u16(uint16_t* out,
                                          const uint16_t* in,
                                          int nrows,
                                          int ncols,
                                          int row_major,
                                          int along_rows,
                                          int total)
{
  reverse2d_kernel(out, in, nrows, ncols, row_major, along_rows, total);
}

extern "C" __global__ void reverse2d_u32(uint32_t* out,
                                          const uint32_t* in,
                                          int nrows,
                                          int ncols,
                                          int row_major,
                                          int along_rows,
                                          int total)
{
  reverse2d_kernel(out, in, nrows, ncols, row_major, along_rows, total);
}

extern "C" __global__ void reverse2d_u64(uint64_t* out,
                                          const uint64_t* in,
                                          int nrows,
                                          int ncols,
                                          int row_major,
                                          int along_rows,
                                          int total)
{
  reverse2d_kernel(out, in, nrows, ncols, row_major, along_rows, total);
}
