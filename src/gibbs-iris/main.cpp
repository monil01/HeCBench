/*---------------------------------------------------------------
  Original author: Zebulun Arendsee
  March 26, 2013
----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "iris_support.hpp"
#include <vector>
#include <hiprand/hiprand_kernel.h>



#define PI 3.14159265359f
#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_ADD 256

/* 
   Box-Muller Transformation: Generate one standard normal variable.

   This algorithm can be more efficiently used by producing two
   random normal variables. However, for the CPU, much faster
   algorithms are possible (e.g. the Ziggurat Algorithm);

   This is actually the algorithm chosen by nVidia to calculate
   normal random variables on the GPU.
 */
float rnorm()
{
  float U1 = rand() / float(RAND_MAX);
  float U2 = rand() / float(RAND_MAX);
  float V1 = sqrtf(-2 * logf(U1)) * cosf(2 * PI * U2);
  return V1;
}

/*
   Generate random gamma variables on a CPU.
 */
float rgamma(float a, float b)
{
  float x;
  if (a > 1) {
    float d = a - 1.f / 3.f;
    float c = 1.f / sqrtf(9.f * d);
    bool flag = true;
    float V;
    while (flag) {
      // Generate a standard normal random variable
      float Z = rnorm();
      if (Z > -1.f / c) {
        V = powf(1.f + c * Z, 3.f);
        float U = rand() / (float)RAND_MAX;
        flag = logf(U) > (0.5f * Z * Z + d - d * V + d * logf(V));
      }
    }
    x = d * V / b;
  }
  else 
  {
    x = rgamma(a + 1.f, b);
    x = x * powf(rand() / (float)RAND_MAX, 1.f / a);
  }
  return x;
}

/*
   Metropolis algorithm for producing random a values. 
   The proposal distribution in normal with a variance that
   is adjusted at each step.
 */
float sample_a(float a, float b, int N, float log_sum)
{
  static float sigma = 2;

  float proposal = rnorm() * sigma + a;

  if(proposal <= 0) return a;

  float log_acceptance_ratio = (proposal - a) * log_sum +
    N * (proposal - a) * logf(b) -
    N * (lgamma(proposal) - lgamma(a));

  float U = rand() / float(RAND_MAX);

  if(logf(U) < log_acceptance_ratio){
    sigma *= 1.1;
    return proposal;
  } else {
    sigma /= 1.1;
    return a;
  }
}

/*
   Returns a random b sample (simply a draw from the appropriate
   gamma distribution)
 */
float sample_b(float a, int N, float flat_sum)
{
  float hyperA = N * a + 1;
  float hyperB = flat_sum;
  return rgamma(hyperA, hyperB);
}

int main(int argc, char *argv[])
{
  const int seed = 123; // seed for random number generation
  srand(seed);

  int trials = 1000; // Default number of trials
  if(argc > 2) {
    trials = atoi(argv[2]);
  }

  /*------ Loading Data ------------------------------------------*/

  FILE *fp;
  if(argc > 1){
    fp = fopen(argv[1], "r");
  } else {
    printf("Please provide input filename\n");
    return 1;
  }

  if(fp == NULL){
    printf("Cannot read file \n");
    return 1;
  }

  int N = 0;
  char line[128];
  while( fgets (line, sizeof line, fp) != NULL ) { 
    N++; 
  }

  rewind(fp);

  size_t y_size = sizeof(int) * N;
  size_t n_size = sizeof(float) * N;
  int *y = (int*) malloc (y_size);
  float *n = (float*) malloc (n_size);
  float *host_fpsum = (float*) malloc (n_size);
  float *host_lpsum = (float*) malloc (n_size);

  for(int i = 0; i < N; i++) {
    fscanf(fp, "%d %f", &y[i], &n[i]);    
  }

  fclose(fp);

  /*------ Memory Allocations ------------------------------------*/
  iris::Platform platform;
  int device = init_iris(platform, argc, argv);
  // IRIS PolicyRandom initialization seeds the process-global C RNG.
  // Restore the original host sampler stream after platform initialization.
  srand(seed);
  int nSumBlocks = (N + THREADS_PER_BLOCK_ADD - 1) / THREADS_PER_BLOCK_ADD;
  std::vector<float> theta(N);
  std::vector<hiprandState> states(N);
  iris::DMem dev_y(y, y_size), dev_n(n, n_size);
  iris::DMem dev_theta(theta.data(), n_size);
  iris::DMem dev_fpsum(host_fpsum,nSumBlocks*sizeof(float));
  iris::DMem dev_lpsum(host_lpsum,nSumBlocks*sizeof(float));
  iris::DMem devStates(states.data(),N*sizeof(hiprandState));
  /*------ Setup RNG ---------------------------------------------*/

  int nBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // Setup the rng machines (one for each thread)
  {
    void* p[] = { &devStates, &N, const_cast<int*>(&seed) };
    int info[] = { iris_w, sizeof(N), sizeof(seed) };
    launch_iris(device, "setup_kernel", nBlocks*THREADS_PER_BLOCK, THREADS_PER_BLOCK, 3, p, info);
  }

  /*------ MCMC Algorithm ----------------------------------------*/
  float a = 20; // Set starting value
  float b = 1;  // Set starting value

  double mean_a = 0.0, mean_b = 0.0;
  double total_time = 0.0;

  for(int i = 0; i < trials; i++) {
    auto start = std::chrono::steady_clock::now();

    {
      void* p[] = { &devStates, &dev_theta, &dev_y, &dev_n, &a, &b, &N };
      int info[] = { iris_rw, iris_w, iris_r, iris_r, sizeof(a), sizeof(b), sizeof(N) };
      launch_iris(device, "sample_theta", nBlocks*THREADS_PER_BLOCK, THREADS_PER_BLOCK, 7, p, info);
    }
    {
      iris::Task task;
      void* p[] = { &dev_theta, &dev_fpsum, &dev_lpsum, &N };
      int info[] = { iris_r, iris_w, iris_w, sizeof(N) };
      size_t global=nSumBlocks*THREADS_PER_BLOCK_ADD, local=THREADS_PER_BLOCK_ADD;
      check_iris(task.kernel("sum_blocks", 1, nullptr, &global, &local, 4, p, info));
      check_iris(task.flush_out(dev_fpsum));
      check_iris(task.flush_out(dev_lpsum));
      check_iris(task.submit(device, nullptr, true));
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time; 

    // The GPU summed blocks of theta values, now the CPU sums the blocks
    float flat_sum = 0;
    float log_sum = 0; 
    for(int j = 0; j < nSumBlocks; j++) {
      flat_sum += host_fpsum[j];
      log_sum += host_lpsum[j];
    }

    // Sample one random value from a's distribution
    a = sample_a(a, b, N, log_sum);
    mean_a += a;

    // And then from b's distribution given the new a
    b = sample_b(a, N, flat_sum);
    mean_b += b;
  }

  printf("Average execution time of kernels: %f (us)\n", (total_time * 1e-3f) / trials);
  printf("a = %lf (avg), b = %lf (avg)\n", mean_a / trials, mean_b / trials);

  /*------ Free Memory -------------------------------------------*/

  free(host_fpsum);
  free(host_lpsum);
  free(y);
  free(n);

  const int errors = platform.error_count();
  check_iris(platform.finalize());
  return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
