#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#if DOUBLE_PRECISION
#define FLOAT double
#else
#define FLOAT float
#endif

typedef struct {
  FLOAT x;
  FLOAT y;
  FLOAT z;
} XYZ;

struct Params {
  int work_group_size;
  const char *file_name;
  int in_size_i;
  int in_size_j;
  int out_size_i;
  int out_size_j;

  Params(int argc, char **argv) {
    work_group_size = 256;
    file_name = "input/control.txt";
    in_size_i = in_size_j = 3;
    out_size_i = out_size_j = 300;
    int opt;
    while ((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:n:")) >= 0) {
      switch (opt) {
        case 'h':
          usage();
          std::exit(0);
        case 'g': work_group_size = std::atoi(optarg); break;
        case 'f': file_name = optarg; break;
        case 'm': in_size_i = in_size_j = std::atoi(optarg); break;
        case 'n': out_size_i = out_size_j = std::atoi(optarg); break;
        default:
          std::fprintf(stderr, "\nUnrecognized option!\n");
          usage();
          std::exit(0);
      }
    }
  }

  void usage() {
    std::fprintf(stderr,
      "\nUsage:  ./main [options]"
      "\n"
      "\nGeneral options:"
      "\n    -h        help"
      "\n    -g <G>    # device work-group size (default=256)"
      "\n"
      "\nBenchmark-specific options:"
      "\n    -f <F>    name of input file with control points (default=input/control.txt)"
      "\n    -m <N>    input size in both dimensions (default=3)"
      "\n    -n <R>    output resolution in both dimensions (default=300)"
      "\n");
  }
};

static void read_input(XYZ *in, const Params &p)
{
  FILE *f = std::fopen(p.file_name, "r");
  if (f == NULL) {
    std::puts("Error opening file");
    std::exit(-1);
  }
  std::printf("Read data from file %s\n", p.file_name);

  int k = 0;
  int ic = 0;
  XYZ v[10000];
#if DOUBLE_PRECISION
  while (std::fscanf(f, "%lf,%lf,%lf", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#else
  while (std::fscanf(f, "%f,%f,%f", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#endif
  {
    ic++;
  }
  std::fclose(f);
  for (int i = 0; i <= p.in_size_i; i++) {
    for (int j = 0; j <= p.in_size_j; j++) {
      in[i * (p.in_size_j + 1) + j] = v[k];
      k = (k + 1) % 16;
    }
  }
}

static int compare_output(XYZ *outp, XYZ *outpCPU, int, int, int RESOLUTIONI, int RESOLUTIONJ)
{
  double sum_delta2 = 0;
  double sum_ref2 = 0;
  for (int i = 0; i < RESOLUTIONI; i++) {
    for (int j = 0; j < RESOLUTIONJ; j++) {
      sum_delta2 += std::fabs(outp[i * RESOLUTIONJ + j].x - outpCPU[i * RESOLUTIONJ + j].x);
      sum_ref2 += std::fabs(outpCPU[i * RESOLUTIONJ + j].x);
      sum_delta2 += std::fabs(outp[i * RESOLUTIONJ + j].y - outpCPU[i * RESOLUTIONJ + j].y);
      sum_ref2 += std::fabs(outpCPU[i * RESOLUTIONJ + j].y);
      sum_delta2 += std::fabs(outp[i * RESOLUTIONJ + j].z - outpCPU[i * RESOLUTIONJ + j].z);
      sum_ref2 += std::fabs(outpCPU[i * RESOLUTIONJ + j].z);
    }
  }
  double L1norm2 = sum_delta2 / sum_ref2;
  if (L1norm2 >= 1e-6) {
    std::printf("Test failed\n");
    return 1;
  }
  return 0;
}

static inline FLOAT BezierBlend(int k, FLOAT mu, int n)
{
  int nn = n;
  int kn = k;
  int nkn = n - k;
  FLOAT blend = 1;
  while (nn >= 1) {
    blend *= nn;
    nn--;
    if (kn > 1) {
      blend /= (FLOAT)kn;
      kn--;
    }
    if (nkn > 1) {
      blend /= (FLOAT)nkn;
      nkn--;
    }
  }
  if (k > 0) {
    blend *= std::pow(mu, (FLOAT)k);
  }
  if (n - k > 0) {
    blend *= std::pow(1 - mu, (FLOAT)(n - k));
  }
  return blend;
}

static void BezierCPU(const XYZ *inp, XYZ *outp, const int NI, const int NJ,
                      const int RESOLUTIONI, const int RESOLUTIONJ)
{
  for (int i = 0; i < RESOLUTIONI; i++) {
    FLOAT mui = i / (FLOAT)(RESOLUTIONI - 1);
    for (int j = 0; j < RESOLUTIONJ; j++) {
      FLOAT muj = j / (FLOAT)(RESOLUTIONJ - 1);
      XYZ out = {0, 0, 0};
      for (int ki = 0; ki <= NI; ki++) {
        FLOAT bi = BezierBlend(ki, mui, NI);
        for (int kj = 0; kj <= NJ; kj++) {
          FLOAT bj = BezierBlend(kj, muj, NJ);
          out.x += inp[ki * (NJ + 1) + kj].x * bi * bj;
          out.y += inp[ki * (NJ + 1) + kj].y * bi * bj;
          out.z += inp[ki * (NJ + 1) + kj].z * bi * bj;
        }
      }
      outp[i * RESOLUTIONJ + j] = out;
    }
  }
}

static int run(XYZ *in, int in_size_i, int in_size_j, int out_size_i, int out_size_j, const Params &p)
{
  size_t out_count = (size_t)out_size_i * out_size_j;
  XYZ *cpu_out = (XYZ*)std::malloc(out_count * sizeof(XYZ));
  XYZ *gpu_out = (XYZ*)std::malloc(out_count * sizeof(XYZ));
  if (!cpu_out || !gpu_out) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(cpu_out);
    std::free(gpu_out);
    return 1;
  }

  auto start = std::chrono::steady_clock::now();
  BezierCPU(in, cpu_out, in_size_i, in_size_j, out_size_i, out_size_j);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "host execution time: " << time << " ms" << std::endl;

  size_t in_size = (size_t)(in_size_i + 1) * (in_size_j + 1) * sizeof(XYZ);
  size_t out_size = out_count * sizeof(XYZ);
  iris::DMem mem_in(in, in_size);
  iris::DMem mem_out(gpu_out, out_size);

  size_t lws[1] = {static_cast<size_t>(p.work_group_size)};
  size_t gws[1] = {((static_cast<size_t>(out_size_i) + lws[0] - 1) / lws[0]) * lws[0]};

  auto kstart = std::chrono::steady_clock::now();
  iris::Task task;
  void *params[6] = {&mem_in, &mem_out, &in_size_i, &in_size_j, &out_size_i, &out_size_j};
  int pinfo[6] = {iris_r, iris_w, sizeof(in_size_i), sizeof(in_size_j), sizeof(out_size_i), sizeof(out_size_j)};
  task.kernel("BezierGPU", 1, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(mem_out);
  task.submit(iris_gpu, nullptr, 1);
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::milliseconds>(kend - kstart).count();
  std::cout << "kernel execution time: " << ktime << " ms" << std::endl;

  mem_out.fetch(gpu_out);
  int status = compare_output(gpu_out, cpu_out, in_size_i, in_size_j, out_size_i, out_size_j);
  std::printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  std::free(cpu_out);
  std::free(gpu_out);
  return status;
}

int main(int argc, char **argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const Params p(argc, argv);
  size_t in_size = (size_t)(p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
  XYZ *h_in = (XYZ*)std::malloc(in_size);
  if (!h_in) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  read_input(h_in, p);
  int status = run(h_in, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j, p);

  std::free(h_in);
  platform.finalize();
  return status == 0 ? 0 : 1;
}
