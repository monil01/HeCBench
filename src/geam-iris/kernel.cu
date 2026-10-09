extern "C" __global__ void transpose_f32_kernel(float* out, const float* in, int nrow, int ncol)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < nrow && col < ncol) {
    out[col * nrow + row] = in[row * ncol + col];
  }
}

extern "C" __global__ void transpose_f64_kernel(double* out, const double* in, int nrow, int ncol)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < nrow && col < ncol) {
    out[col * nrow + row] = in[row * ncol + col];
  }
}
