extern "C" __global__ void dev_lapl_iter(float* out, const float* in, const float delta,
                                          const float norm, const int Lx, const int Ly)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = Lx * Ly;
  if (i >= n) {
    return;
  }

  const int x = i % Lx;
  const int y = i / Lx;
  const int v00 = y * Lx + x;
  const int v0p = y * Lx + (x + 1) % Lx;
  const int v0m = y * Lx + (Lx + x - 1) % Lx;
  const int vp0 = ((y + 1) % Ly) * Lx + x;
  const int vm0 = ((Ly + y - 1) % Ly) * Lx + x;
  out[v00] = norm * in[v00] + delta * (in[v0p] + in[v0m] + in[vp0] + in[vm0]);
}
