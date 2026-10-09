#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHUNK_S 4096

typedef struct {
  float x, y, z;
} kdata;

void RunFHd(float* h_rmu, float* h_imu, kdata* h_k, float* h_rfhd,
            float* h_ifhd, float* h_x, float* h_y, float* h_z, int samples,
            int voxels) {
  const int ntpb = 256;
  size_t gws = static_cast<size_t>((samples + ntpb - 1) / ntpb) * ntpb;
  size_t lws = ntpb;
  const int sampleSize = samples * sizeof(float);

  iris::DMem mem_rfhd(h_rfhd, sampleSize);
  iris::DMem mem_ifhd(h_ifhd, sampleSize);
  iris::DMem mem_x(h_x, sampleSize);
  iris::DMem mem_y(h_y, sampleSize);
  iris::DMem mem_z(h_z, sampleSize);

  int nchunks = (voxels + CHUNK_S - 1) / CHUNK_S;

  for (int i = 0; i < nchunks; i++) {
    int c = CHUNK_S;
    if (i == nchunks - 1) c = voxels - CHUNK_S * i;
    const int chunkFloatSize = c * sizeof(float);
    const int chunkKSize = c * sizeof(kdata);

    iris::DMem mem_rmu(h_rmu + i * CHUNK_S, chunkFloatSize);
    iris::DMem mem_imu(h_imu + i * CHUNK_S, chunkFloatSize);
    iris::DMem mem_k(h_k + i * CHUNK_S, chunkKSize);

    int samples_arg = samples;
    int voxels_arg = c;
    iris::Task task;
    void* params[10] = {&mem_k,    &mem_rmu,  &mem_imu, &mem_rfhd, &mem_ifhd,
                        &mem_x,    &mem_y,    &mem_z,   &samples_arg,
                        &voxels_arg};
    int pinfo[10] = {iris_r, iris_r, iris_r,  iris_rw, iris_rw,
                     iris_r, iris_r, iris_r,  sizeof(samples_arg),
                     sizeof(voxels_arg)};
    task.kernel("cmpfhd", 1, nullptr, &gws, &lws, 10, params, pinfo);
    if (i == nchunks - 1) {
      task.flush_out(mem_rfhd);
      task.flush_out(mem_ifhd);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::printf("Usage: %s <#samples> <#voxels> <verify>\n", argv[0]);
    std::exit(1);
  }
  const int samples = std::atoi(argv[1]);
  const int voxels = std::atoi(argv[2]);
  const int verify = std::atoi(argv[3]);
  const int sampleSize = samples * sizeof(float);
  const int voxelSize = voxels * sizeof(float);

  float* h_rmu = static_cast<float*>(std::malloc(voxelSize));
  float* h_imu = static_cast<float*>(std::malloc(voxelSize));
  float* h_kx = static_cast<float*>(std::malloc(voxelSize));
  float* h_ky = static_cast<float*>(std::malloc(voxelSize));
  float* h_kz = static_cast<float*>(std::malloc(voxelSize));
  kdata* h_k = static_cast<kdata*>(std::malloc(voxels * sizeof(kdata)));

  float* h_rfhd = static_cast<float*>(std::malloc(sampleSize));
  float* h_ifhd = static_cast<float*>(std::malloc(sampleSize));
  float* h_x = static_cast<float*>(std::malloc(sampleSize));
  float* h_y = static_cast<float*>(std::malloc(sampleSize));
  float* h_z = static_cast<float*>(std::malloc(sampleSize));

  float* rfhd = static_cast<float*>(std::malloc(sampleSize));
  float* ifhd = static_cast<float*>(std::malloc(sampleSize));

  std::srand(2);
  for (int i = 0; i < samples; i++) {
    h_rfhd[i] = static_cast<float>(i) / samples;
    h_ifhd[i] = static_cast<float>(i) / samples;
    h_x[i] = 0.3f + (std::rand() % 2 ? 0.1f : -0.1f);
    h_y[i] = 0.2f + (std::rand() % 2 ? 0.1f : -0.1f);
    h_z[i] = 0.1f + (std::rand() % 2 ? 0.1f : -0.1f);
  }

  for (int i = 0; i < voxels; i++) {
    h_rmu[i] = static_cast<float>(i) / voxels;
    h_imu[i] = static_cast<float>(i) / voxels;
    h_k[i].x = h_kx[i] = 0.1f + (std::rand() % 2 ? 0.1f : -0.1f);
    h_k[i].y = h_ky[i] = 0.2f + (std::rand() % 2 ? 0.1f : -0.1f);
    h_k[i].z = h_kz[i] = 0.3f + (std::rand() % 2 ? 0.1f : -0.1f);
  }

  std::printf("Run FHd on a device\n");

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  auto start = std::chrono::steady_clock::now();
  RunFHd(h_rmu, h_imu, h_k, h_rfhd, h_ifhd, h_x, h_y, h_z, samples, voxels);
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Device execution time %f (s)\n", time * 1e-9f);

  for (int i = 0; i < samples; i++) {
    rfhd[i] = h_rfhd[i];
    ifhd[i] = h_ifhd[i];
  }

  platform.finalize();

  if (verify) {
    std::printf("Computing root mean square error between host and device "
                "results.\n");
    std::printf("This will take a while..\n");

#pragma omp parallel for
    for (int n = 0; n < samples; n++) {
      float r = static_cast<float>(n) / samples;
      float i = static_cast<float>(n) / samples;
#pragma omp parallel for simd reduction(+ : r, i)
      for (int m = 0; m < voxels; m++) {
        float e = 2.f * static_cast<float>(M_PI) *
                  (h_kx[m] * h_x[n] + h_ky[m] * h_y[n] + h_kz[m] * h_z[n]);
        float c = std::cos(e);
        float s = std::sin(e);
        r += h_rmu[m] * c - h_imu[m] * s;
        i += h_imu[m] * c + h_rmu[m] * s;
      }
      h_rfhd[n] = r;
      h_ifhd[n] = i;
    }

    float err = 0.f;
    for (int i = 0; i < samples; i++) {
      err += (h_rfhd[i] - rfhd[i]) * (h_rfhd[i] - rfhd[i]) +
             (h_ifhd[i] - ifhd[i]) * (h_ifhd[i] - ifhd[i]);
    }
    std::printf("RMSE = %f\n", std::sqrt(err / (2 * samples)));
  }

  std::free(h_rmu);
  std::free(h_imu);
  std::free(h_kx);
  std::free(h_ky);
  std::free(h_kz);
  std::free(h_k);
  std::free(h_rfhd);
  std::free(h_ifhd);
  std::free(rfhd);
  std::free(ifhd);
  std::free(h_x);
  std::free(h_y);
  std::free(h_z);

  return 0;
}
