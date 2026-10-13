#include <math.h>
#include <float.h>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
static void check(int rc) { if (rc != IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc); exit(2); } }

#include "reference.h"

using namespace std::chrono;
typedef high_resolution_clock myclock;
typedef duration<float> myduration;

#define MAX_WG_SIZE 256

template <typename T>
T* mem_alloc (const int align, const size_t size) {
  return (T*) aligned_alloc(align, size * sizeof(T));
}

template <typename T>
void mem_free (T* p) {
  free(p);
}

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of samples> <number of SNPs> <repeat>\n", argv[0]);
    return 1;
  }
  int i, j, x;
  int num_pac = atoi(argv[1]);  // #samples
  int num_snp = atoi(argv[2]);  // #SNPs
  int iteration = atoi(argv[3]);// #kernel run
  int block_snp = 64;

  srand(100);
  unsigned char *SNP_Data = mem_alloc<unsigned char>(64, num_pac * num_snp);
  unsigned char *Ph_Data = mem_alloc<unsigned char>(64, num_pac);

  // generate SNPs between 0 and 2
  for (i = 0; i < num_pac; i++)
    for(j = 0; j < num_snp; j++)
      SNP_Data[i * num_snp + j] = rand() % 3;

  // generate phenotype between 0 and 1
  for(int i = 0; i < num_pac; i++) Ph_Data[i] = rand() % 2;

  // transpose the SNP data
  unsigned char *SNP_Data_trans = mem_alloc<unsigned char>(64, num_pac * num_snp);

  for (i = 0; i < num_pac; i++) 
    for(j = 0; j < num_snp; j++) 
      SNP_Data_trans[j * num_pac + i] = SNP_Data[i * num_snp + j];

  int phen_ones = 0;
  for(i = 0; i < num_pac; i++)
    if(Ph_Data[i] == 1)
      phen_ones++;

  // transform SNP data to a binary format

  int PP_zeros = ceil((1.0*(num_pac - phen_ones))/32.0);
  int PP_ones = ceil((1.0*phen_ones)/32.0);

  unsigned int *bin_data_zeros = mem_alloc<unsigned int>(64, num_snp * PP_zeros * 2);
  unsigned int *bin_data_ones = mem_alloc<unsigned int>(64, num_snp * PP_ones * 2);
  memset(bin_data_zeros, 0, num_snp*PP_zeros*2*sizeof(unsigned int));
  memset(bin_data_ones, 0, num_snp*PP_ones*2*sizeof(unsigned int));

  for(i = 0; i < num_snp; i++)
  {
    int x_zeros = -1;
    int x_ones = -1;
    int n_zeros = 0;
    int n_ones = 0;

    for(j = 0; j < num_pac; j++){
      unsigned int temp = (unsigned int) SNP_Data_trans[i * num_pac + j];

      if(Ph_Data[j] == 1){
        if(n_ones%32 == 0){
          x_ones ++;
        }
        // apply 1 shift left to 2 components
        bin_data_ones[i * PP_ones * 2 + x_ones*2 + 0] <<= 1;
        bin_data_ones[i * PP_ones * 2 + x_ones*2 + 1] <<= 1;
        // insert '1' in correct component
        if(temp == 0 || temp == 1){
          bin_data_ones[i * PP_ones * 2 + x_ones*2 + temp ] |= 1;
        }
        n_ones ++;
      } else {
        if(n_zeros%32 == 0){
          x_zeros ++;
        }
        // apply 1 shift left to 2 components
        bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 0] <<= 1;
        bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 1] <<= 1;
        // insert '1' in correct component
        if(temp == 0 || temp == 1){
          bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + temp] |= 1;
        }
        n_zeros ++;
      }
    }
  }

  unsigned int mask_zeros = 0xFFFFFFFF;
  for(int x = num_pac - phen_ones; x < PP_zeros * 32; x++)
    mask_zeros = mask_zeros >> 1;

  unsigned int mask_ones = 0xFFFFFFFF;
  for(x = phen_ones; x < PP_ones * 32; x++)
    mask_ones = mask_ones >> 1;

  // transpose the binary data structures
  unsigned int* bin_data_ones_trans = mem_alloc<unsigned int>(64, num_snp * PP_ones * 2);

  for(i = 0; i < num_snp; i++)
    for(j = 0; j < PP_ones; j++)
    {
      bin_data_ones_trans[(j * num_snp + i) * 2 + 0] = bin_data_ones[(i * PP_ones + j) * 2 + 0];
      bin_data_ones_trans[(j * num_snp + i) * 2 + 1] = bin_data_ones[(i * PP_ones + j) * 2 + 1];
    }

  unsigned int* bin_data_zeros_trans = mem_alloc<unsigned int>(64, num_snp * PP_zeros * 2);

  for(i = 0; i < num_snp; i++)
    for(j = 0; j < PP_zeros; j++)
    {
      bin_data_zeros_trans[(j * num_snp + i) * 2 + 0] = bin_data_zeros[(i * PP_zeros + j) * 2 + 0];
      bin_data_zeros_trans[(j * num_snp + i) * 2 + 1] = bin_data_zeros[(i * PP_zeros + j) * 2 + 1];
    }

  float *scores = mem_alloc<float>(64, num_snp * num_snp);
  float *scores_ref = mem_alloc<float>(64, num_snp * num_snp);
  for(x = 0; x < num_snp * num_snp; x++) {
    scores[x] = scores_ref[x] = FLT_MAX;
  }

  iris::Platform platform;
  check(platform.init(&argc, &argv, true));
  platform.overview();
  iris::DMem d_data_zeros(bin_data_zeros_trans, num_snp*PP_zeros*2*sizeof(unsigned int));
  iris::DMem d_data_ones(bin_data_ones_trans, num_snp*PP_ones*2*sizeof(unsigned int));
  iris::DMem d_scores(scores, sizeof(float)*num_snp*num_snp);
  // setup kernel ND-range
  int num_snp_m = num_snp;
  while(num_snp_m % block_snp != 0) num_snp_m++;

  size_t gws[] = {size_t(num_snp_m), size_t(num_snp_m)};
  size_t lws[] = {size_t(block_snp), 1};

  // epistasis detection kernel

  auto kstart = myclock::now();

  for (int i = 0; i < iteration; i++) {
    iris::Task task;
    void* params[] = {&d_data_zeros, &d_data_ones, &d_scores, &num_snp,
                      &PP_zeros, &PP_ones, &mask_zeros, &mask_ones};
    int access[] = {iris_r, iris_r, iris_rw, sizeof(int), sizeof(int),
                    sizeof(int), sizeof(unsigned int), sizeof(unsigned int)};
    check(task.kernel("epi", 2, nullptr, gws, lws, 8, params, access));
    check(task.submit(iris_amd, nullptr, true));
  }


  myduration ktime = myclock::now() - kstart;
  auto total_ktime = ktime.count();

  std::cout << "Average kernel execution time: "
            << total_ktime / iteration << " (s)" << std::endl;

  iris::Task flush;
  check(flush.flush_out(d_scores));
  check(flush.submit(iris_amd, nullptr, true));

  int p1 = min_score(scores, num_snp, num_snp);

  reference (bin_data_zeros_trans, bin_data_ones_trans, scores_ref, num_snp, 
             PP_zeros, PP_ones, mask_zeros, mask_ones);

  int p2 = min_score(scores_ref, num_snp, num_snp);
  
  bool ok = (p1 == p2) && (fabsf(scores[p1] - scores_ref[p2]) < 1e-3f);
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;

  mem_free(bin_data_zeros);
  mem_free(bin_data_ones);
  mem_free(bin_data_zeros_trans);
  mem_free(bin_data_ones_trans);
  mem_free(scores);
  mem_free(scores_ref);
  mem_free(SNP_Data);
  mem_free(SNP_Data_trans);
  mem_free(Ph_Data);
  return 0;
}
