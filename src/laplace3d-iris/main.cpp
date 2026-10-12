#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int BLOCK_X = 32;
static constexpr int BLOCK_Y = 8;

static void reference(int nx, int ny, int nz, const float* u1, float* u2)
{
  constexpr float sixth = 1.0f / 6.0f;
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        const int ind = i + j * nx + k * nx * ny;
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
          u2[ind] = u1[ind];
        } else {
          u2[ind] = (u1[ind - 1] + u1[ind + 1] +
                     u1[ind - nx] + u1[ind + nx] +
                     u1[ind - nx * ny] + u1[ind + nx * ny]) * sixth;
        }
      }
    }
  }
}

static void launch_laplace(iris::DMem& u1,
                           iris::DMem& u2,
                           int nx,
                           int ny,
                           int nz,
                           int pitch)
{
  const int bx = 1 + (nx - 1) / BLOCK_X;
  const int by = 1 + (ny - 1) / BLOCK_Y;
  size_t gws[2] = {static_cast<size_t>(bx * BLOCK_X), static_cast<size_t>(by * BLOCK_Y)};
  size_t lws[2] = {BLOCK_X, BLOCK_Y};
  void* params[] = {&nx, &ny, &nz, &pitch, &u1, &u2};
  int pinfo[] = {sizeof(nx), sizeof(ny), sizeof(nz), sizeof(pitch), iris_r, iris_w};
  iris::Task task;
  task.kernel("laplace3d", 2, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(u2);
  task.submit(iris_gpu, nullptr, 1);
}

static void print_help()
{
  std::printf("Usage:  laplace3d [OPTION]...\n");
  std::printf("6-point stencil 3D Laplace test \n\n");
  std::printf("Example: run 100 iterations on a 256x128x128 grid\n");
  std::printf("./main 256 128 128 100 1\n\n");
  std::printf("Options:\n");
  std::printf("Grid width\n");
  std::printf("Grid height\n");
  std::printf("Grid depth\n");
  std::printf("Number of repetitions\n");
  std::printf("verify the result\n");
}

int main(int argc, char** argv)
{
  if (argc != 6) {
    print_help();
    return 1;
  }

  const int nx = std::atoi(argv[1]);
  const int ny = std::atoi(argv[2]);
  const int nz = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  const int verify = std::atoi(argv[5]);

  if (nx <= 0 || nx % 32 != 0 || ny <= 0 || nz <= 0 || repeat <= 0) return 1;

  std::printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);
  std::printf("Result verification %s \n", verify ? "enabled" : "disabled");

  const size_t grid3d_size = static_cast<size_t>(nx) * ny * nz;
  std::vector<float> h_u1(grid3d_size);
  std::vector<float> h_u2(grid3d_size);
  std::vector<float> h_u3(grid3d_size);

  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        const int ind = i + j * nx + k * nx * ny;
        h_u1[ind] = (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
            ? 1.0f
            : 0.0f;
      }
    }
  }
  std::vector<float> h_ref = h_u1;

  const int pitch = nx;
  const int bx = 1 + (nx - 1) / BLOCK_X;
  const int by = 1 + (ny - 1) / BLOCK_Y;
  std::printf("\ndimGrid  = %d %d %d \n", bx, by, 1);
  std::printf("dimBlock = %d %d %d \n", BLOCK_X, BLOCK_Y, 1);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  iris::DMem mem_u1(h_u1.data(), h_u1.size() * sizeof(float));
  iris::DMem mem_u2(h_u2.data(), h_u2.size() * sizeof(float));

  launch_laplace(mem_u1, mem_u2, nx, ny, nz, pitch);

  iris::DMem* src = &mem_u1;
  iris::DMem* dst = &mem_u2;
  auto start = std::chrono::steady_clock::now();
  for (int i = 1; i <= repeat; ++i) {
    launch_laplace(*src, *dst, nx, ny, nz, pitch);
    if (dst == &mem_u1) {
      mem_u1.fetch(h_u1.data());
    } else {
      mem_u2.fetch(h_u2.data());
    }
    std::swap(src, dst);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  src->fetch(h_u2.data());

  if (verify) {
    for (int i = 1; i <= repeat; ++i) {
      reference(nx, ny, nz, h_ref.data(), h_u3.data());
      std::swap(h_ref, h_u3);
    }

    bool ok = true;
    float err = 0.0f;
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          const int ind = i + j * nx + k * nx * ny;
          err += (h_ref[ind] - h_u2[ind]) * (h_ref[ind] - h_u2[ind]);
          if (std::fabs(h_ref[ind] - h_u2[ind]) > 1e-3f) ok = false;
        }
      }
    }
    std::printf("\n RMS error = %f \n", std::sqrt(err / nx * ny * nz));
    std::printf(" %s\n", ok ? "PASS" : "FAIL");
    platform.finalize();
    return ok ? 0 : 2;
  }

  platform.finalize();
  return 0;
}
