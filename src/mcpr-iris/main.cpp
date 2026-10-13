#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include "iris_support.h"
#include "reference.h"

// transpose
double* t(const double *idata, const int width, const int height)
{
  double *odata = (double*) malloc (sizeof(double) * width * height);
  for (int yIndex = 0; yIndex < height; yIndex++) {
    for (int xIndex = 0; xIndex < width; xIndex++) {
      int index_in  = xIndex + width * yIndex;
      int index_out = yIndex + height * xIndex;
      odata[index_out] = idata[index_in];
    }
  }
  return odata;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <path to filename> <repeat>\n", argv[0]);
    return 1;
  }
  char *filename = argv[1];
  const int repeat = atoi(argv[2]);
  if (repeat <= 0) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc, &argv, true), "init");
  const int device = iris_hip_device();

  // n and K should match the dimension of the dataset in the csv file
  const int n = 26280, K = 21, M = 10000;

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: failed to open file alphas.csv. Exit\n");
    return 1;
  }

  int alphas_size = n * K; // n rows and K cols
  int alphas_size_byte = n * K * sizeof(double);

  int rands_size = M * K;  // M rows and K cols
  int rands_size_byte = M * K * sizeof(double);

  double *alphas, *rands, *probs, *probs_ref;
  alphas = (double*) malloc (alphas_size_byte);
  rands = (double*) malloc (rands_size_byte);
  probs = (double*) malloc (alphas_size_byte);
  probs_ref = (double*) malloc (alphas_size_byte);

  // load the csv file
  for (int i = 0; i < alphas_size; i++)
    fscanf(fp, "%lf", &alphas[i]);
  fclose(fp);

  // normal distribution (mean: 0 and var: 1)
  std::mt19937 gen(19937);
  std::normal_distribution<double> norm_dist(0.0,1.0);
  for (int i = 0; i < rands_size; i++) rands[i] = norm_dist(gen);

  reference(alphas, rands, probs_ref, n, K, M);

  iris::DMem d_alphas(alphas,alphas_size_byte);
  iris::DMem d_rands(rands,rands_size_byte);
  iris::DMem d_probs(probs,alphas_size_byte);
  auto launch = [&](const char* name, int blocksize) {
    size_t lws=blocksize, gws=((n+blocksize-1)/blocksize)*blocksize;
    iris::Task task;
    void* args[]={&d_alphas,&d_rands,&d_probs,(void*)&n,(void*)&K,(void*)&M};
    int info[]={iris_r,iris_r,iris_w,sizeof(n),sizeof(K),sizeof(M)};
    iris_check(task.kernel(name,1,nullptr,&gws,&lws,6,args,info),"kernel");
    iris_check(task.flush_out(d_probs),"flush probs");
    iris_check(task.submit(device,nullptr,true),"HIP task");
  };

  // kernel 1
  int threads_per_block = 192;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    launch("compute_probs", threads_per_block);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of compute_probs kernel: %f (s)\n", (time * 1e-9f) / repeat);

  verify(probs, probs_ref, alphas_size);

  // kernel 2
  double *t_rands = t(rands, K, M);
  double *t_alphas = t(alphas, K, n);

  reference_unitStrides(t_alphas, t_rands, probs_ref, n, K, M);

  iris_check(d_rands.update(t_rands), "refresh transposed rands");
  iris_check(d_alphas.update(t_alphas), "refresh transposed alphas");

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    launch("compute_probs_unitStrides", threads_per_block);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of compute_probs_unitStrides kernel: %f (s)\n", (time * 1e-9f) / repeat);

  verify(probs, probs_ref, alphas_size);

  // kernel 3
  threads_per_block = 96;

  // The fixed K=21, block=96 shared storage is declared statically in the backend.

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    launch("compute_probs_unitStrides_sharedMem", threads_per_block);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of compute_probs_unitStrides_sharedMem kernel: %f (s)\n", (time * 1e-9f) / repeat);

  verify(probs, probs_ref, alphas_size);

  // free memory
  free(alphas);
  free(rands);
  free(t_alphas);
  free(t_rands);
  free(probs);
  free(probs_ref);
  iris_check(platform.finalize(),"finalize");
  return verification_errors ? 1 : 0;
}
