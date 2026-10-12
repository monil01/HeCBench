#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

static void init(bool is_e4m3, float* src, int nelems) {
  std::mt19937 gen(19937);
  float min = is_e4m3 ? -240.0f : -57344.0f;
  float max = is_e4m3 ? 240.0f : 57344.0f;
  std::uniform_real_distribution<float> dis(min, max + 1.0f);

  if (is_e4m3) {
    src[0] = powf(2.0f, -10.0f);
    src[1] = 0.875f * src[0];
    src[2] = powf(2.0f, -11.0f);
    src[3] = 240.0f;
    src[4] = 448.0f;
  } else {
    src[0] = powf(2.0f, -17.0f);
    src[1] = 0.75f * src[0];
    src[2] = powf(2.0f, -18.0f);
    src[3] = 57344.0f;
    src[4] = 57345.0f;
  }
  src[5] = INFINITY;
  src[6] = -INFINITY;
  src[7] = NAN;
  for (int i = 8; i < nelems; ++i) {
    src[i] = dis(gen);
  }
}

static double launch_kernel(const char* kernel_name, uint8_t* dst, float* src,
                            int nelems, int niters) {
  constexpr int block_size = 256;
  size_t gws[] = {static_cast<size_t>(nelems)};
  size_t lws[] = {block_size};
  iris::DMem mem_src(src, static_cast<size_t>(nelems) * sizeof(float));
  iris::DMem mem_dst(dst, static_cast<size_t>(nelems) * sizeof(uint8_t));

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; ++i) {
    iris::Task task;
    void* params[] = {&mem_dst, &mem_src, &nelems};
    int pinfo[] = {iris_w, iris_r, static_cast<int>(sizeof(nelems))};
    task.kernel(kernel_name, 1, nullptr, gws, lws, 3, params, pinfo);
    task.flush_out(mem_dst);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
         1.0 / niters;
}

static void convert(bool is_e4m3, int nelems, int niters) {
  float* h_src = static_cast<float*>(malloc(static_cast<size_t>(nelems) * sizeof(float)));
  uint8_t* h_dst =
      static_cast<uint8_t*>(malloc(static_cast<size_t>(nelems) * sizeof(uint8_t)));
  uint8_t* r_dst =
      static_cast<uint8_t*>(malloc(static_cast<size_t>(nelems) * sizeof(uint8_t)));
  if (!h_src || !h_dst || !r_dst) {
    fprintf(stderr, "host allocation failed\n");
    exit(1);
  }

  init(is_e4m3, h_src, nelems);

  const char* ref_kernel =
      is_e4m3 ? "ref_fp32_cvt_e4m3fnuz_kernel" : "ref_fp32_cvt_e5m2fnuz_kernel";
  const char* intrinsic_kernel =
      is_e4m3 ? "fp32_cvt_e4m3_kernel" : "fp32_cvt_e5m2_kernel";

  double time = launch_kernel(ref_kernel, r_dst, h_src, nelems, niters);
  double size = static_cast<double>(sizeof(uint8_t) + sizeof(float)) * nelems;
  printf("size(GB):%.2f, average time of reference (sec):%f, BW:%f\n", size,
         time * 1e-9, size / time);

  time = launch_kernel(intrinsic_kernel, h_dst, h_src, nelems, niters);
  printf("size(GB):%.2f, average time of intrinsic (sec):%f, BW:%f\n", size,
         time * 1e-9, size / time);

  bool ok = true;
  int mismatch = -1;
  for (int i = 0; i < nelems; ++i) {
    if (abs(static_cast<int8_t>(h_dst[i] - r_dst[i])) > 1) {
      ok = false;
      mismatch = i;
      break;
    }
  }
  if (!ok) {
    printf("first mismatch at %d: intrinsic=%u reference=%u src=%f\n", mismatch,
           static_cast<unsigned>(h_dst[mismatch]), static_cast<unsigned>(r_dst[mismatch]),
           h_src[mismatch]);
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_src);
  free(h_dst);
  free(r_dst);
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }

  const int nelems = atoi(argv[1]);
  const int niters = atoi(argv[2]);
  if (nelems < 8 || niters <= 0) {
    fprintf(stderr, "number of elements must be at least 8 and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  printf("float -> fp8 E4M3FNUZ\n");
  convert(true, nelems, niters);

  printf("float -> fp8 E5M2FNUZ\n");
  convert(false, nelems, niters);

  platform.finalize();
  return 0;
}
