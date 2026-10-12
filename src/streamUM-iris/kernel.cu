extern "C" __global__ void streamum_dgemv_col_major(
    double* result, const double* data, const double* vector,
    const unsigned int size)
{
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= size) return;

  double sum = 0.0;
  for (unsigned int row = 0; row < size; row++) {
    sum += data[row * size + col] * vector[row];
  }
  result[col] = sum;
}
