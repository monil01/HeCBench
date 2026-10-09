extern "C" __global__ void glu_kernel(
    const int M,
    const int split_dim_size,
    const int N,
    const float* Xdata,
    float* Ydata)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= M * split_dim_size * N) return;

  const int xOffset = 2 * split_dim_size * N;
  const int yOffset = split_dim_size * N;
  const int i = index / split_dim_size / N;
  const int j = index / N % split_dim_size;
  const int k = index % N;
  const float x1 = Xdata[i * xOffset + j * N + k];
  const float x2 = Xdata[i * xOffset + (j + split_dim_size) * N + k];
  Ydata[i * yOffset + j * N + k] = x1 * (1.f / (1.f + expf(-x2)));
}
