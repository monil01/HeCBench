#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#define NTX 16
#define NTY 16

static void usage(char* argv[])
{
  std::fprintf(stderr, " Usage: %s LX LY NITER\n", argv[0]);
}

static void reference(float* out, const float* in, const float delta, const float norm,
                      const int Lx, const int Ly)
{
#pragma omp parallel for collapse(2)
  for (int y = 0; y < Ly; y++) {
    for (int x = 0; x < Lx; x++) {
      const int v00 = y * Lx + x;
      const int v0p = y * Lx + (x + 1) % Lx;
      const int v0m = y * Lx + (Lx + x - 1) % Lx;
      const int vp0 = ((y + 1) % Ly) * Lx + x;
      const int vm0 = ((Ly + y - 1) % Ly) * Lx + x;
      out[v00] = norm * in[v00] + delta * (in[v0p] + in[v0m] + in[vp0] + in[vm0]);
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    usage(argv);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int Lx = std::atoi(argv[1]);
  const int Ly = std::atoi(argv[2]);
  if (Lx % NTX != 0 || Ly % NTY != 0) {
    std::printf("Array length LX and LY must be a multiple of block size %d and %d, respectively\n",
                NTX, NTY);
    platform.finalize();
    return 1;
  }

  const int niter = std::atoi(argv[3]);
  assert(niter >= 1);

  const float sigma = 0.01f;
  const float xdelta = sigma / (1.0f + 4.0f * sigma);
  const float xnorm = 1.0f / (1.0f + 4.0f * sigma);

  std::printf(" Ly,Lx = %d,%d\n", Ly, Lx);
  std::printf(" niter = %d\n", niter);

  const size_t nelems = static_cast<size_t>(Lx) * static_cast<size_t>(Ly);
  const size_t bytes = nelems * sizeof(float);

  std::srand(123);
  float* buffer = static_cast<float*>(std::malloc(bytes));
  float* h_out = static_cast<float*>(std::malloc(bytes));
  float* h_in = static_cast<float*>(std::malloc(bytes));
  float* d_res = static_cast<float*>(std::malloc(bytes));
  float* iris_tmp = static_cast<float*>(std::malloc(bytes));
  if (!buffer || !h_out || !h_in || !d_res || !iris_tmp) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(buffer);
    std::free(h_out);
    std::free(h_in);
    std::free(d_res);
    std::free(iris_tmp);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < Lx; i += 16) {
    const int x = std::rand() % Lx;
    for (int j = 0; j < Ly; j++) {
      buffer[x + j * Lx] = 1.0f;
    }
  }
  for (int i = 0; i < Ly; i += 16) {
    const int y = std::rand() % Ly;
    for (int j = 0; j < Lx; j++) {
      buffer[j + y * Lx] = 1.0f;
    }
  }
  std::memcpy(h_in, buffer, bytes);
  std::memcpy(d_res, buffer, bytes);

  for (int i = 0; i < niter; i++) {
    reference(h_out, h_in, xdelta, xnorm, Lx, Ly);
    std::swap(h_out, h_in);
  }

  iris::DMem mem_a(d_res, bytes);
  iris::DMem mem_b(iris_tmp, bytes);
  iris::DMem* mem_in = &mem_a;
  iris::DMem* mem_out = &mem_b;

  const size_t gws[1] = { nelems };
  const size_t lws[1] = { NTX * NTY };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < niter; i++) {
    iris::Task task;
    void* params[6] = { mem_out, mem_in, const_cast<float*>(&xdelta),
                        const_cast<float*>(&xnorm), const_cast<int*>(&Lx),
                        const_cast<int*>(&Ly) };
    int pinfo[6] = { iris_w, iris_r, sizeof(xdelta), sizeof(xnorm), sizeof(Lx), sizeof(Ly) };
    task.kernel("dev_lapl_iter", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 6, params, pinfo);
    if (i == niter - 1) {
      task.flush_out(*mem_out);
    }
    task.submit(iris_gpu, nullptr, 1);
    std::swap(mem_in, mem_out);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const double t0 = time * 1e-3 / niter;

  std::printf("Device: iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
              niter, Lx, Ly, t0, nelems * sizeof(float) * 2.0 / (t0 * 1.0e3),
              (nelems * 6.0) / (t0 * 1.0e3));

  const float* result = (niter % 2 == 0) ? d_res : iris_tmp;
  bool ok = true;
  for (size_t i = 0; i < nelems; i++) {
    if (std::fabs(h_in[i] - result[i]) > 1e-2f) {
      std::printf("Mismatch at %zu cpu=%f gpu=%f\n", i, h_in[i], result[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(buffer);
  std::free(h_out);
  std::free(h_in);
  std::free(d_res);
  std::free(iris_tmp);
  platform.finalize();
  return ok ? 0 : 1;
}
