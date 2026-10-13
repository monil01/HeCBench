// ************************************************
// original authors: Lee Howes and David B. Thomas
// ************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "iris_support.hpp"
#include "loopback.h"


int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dump> <repeat>\n", argv[0]);
    return 1;
  }
  // display device results when enabled
  const int dump = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t loopback_size = sizeof(float) * LOOKBACK_NUM_PARAMETER_VALUES;
  const size_t seed_size = sizeof(unsigned int) * TAUSWORTHE_NUM_SEEDS;

  float *lookback_VOL_0 = (float *) malloc(loopback_size);
  float *lookback_A_0 = (float *) malloc(loopback_size);
  float *lookback_A_1 = (float *) malloc(loopback_size);
  float *lookback_A_2 = (float *) malloc(loopback_size);
  float *lookback_S_0 = (float *) malloc(loopback_size);
  float *lookback_EPS_0 = (float *) malloc(loopback_size);
  float *lookback_MU = (float *) malloc(loopback_size);
  float *lookbackSimulationResultsMean = (float *) malloc(loopback_size);
  float *lookbackSimulationResultsVariance = (float *) malloc(loopback_size);

  for (unsigned i = 0; i < LOOKBACK_NUM_PARAMETER_VALUES; i++)
  {
    lookback_VOL_0[i] = Rand();
    lookback_A_0[i] = Rand();
    lookback_A_1[i] = Rand();
    lookback_A_2[i] = Rand();
    lookback_S_0[i] = Rand();
    lookback_EPS_0[i] = Rand();
    lookback_MU[i] = Rand();
  }

  unsigned int *tauswortheSeeds = (unsigned int *) malloc(seed_size);
  for (unsigned i = 0; i < TAUSWORTHE_NUM_SEEDS; i++)
    tauswortheSeeds[i] = (unsigned int)rand() + 16;

  iris::Platform platform;
  int device=init_iris(platform,argc,argv);
  iris::DMem d_tauswortheSeeds(tauswortheSeeds,seed_size);
  iris::DMem d_lookback_VOL_0(lookback_VOL_0,loopback_size);
  iris::DMem d_lookback_A_0(lookback_A_0,loopback_size),d_lookback_A_1(lookback_A_1,loopback_size);
  iris::DMem d_lookback_A_2(lookback_A_2,loopback_size),d_lookback_S_0(lookback_S_0,loopback_size);
  iris::DMem d_lookback_EPS_0(lookback_EPS_0,loopback_size),d_lookback_MU(lookback_MU,loopback_size);
  iris::DMem d_lookbackSimulationResultsMean(lookbackSimulationResultsMean,loopback_size);
  iris::DMem d_lookbackSimulationResultsVariance(lookbackSimulationResultsVariance,loopback_size);
  int blocks=getenv("LOOPBACK_VALIDATION_BLOCKS")?atoi(getenv("LOOPBACK_VALIDATION_BLOCKS")):LOOKBACK_TAUSWORTHE_NUM_BLOCKS;
  if(blocks<1||blocks>LOOKBACK_TAUSWORTHE_NUM_BLOCKS) return 1;
  unsigned active_count=blocks*LOOKBACK_TAUSWORTHE_NUM_THREADS;
  const unsigned num_cycles = LOOKBACK_MAX_T;


  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    void* p[]={const_cast<unsigned*>(&num_cycles),&d_tauswortheSeeds,
       &d_lookbackSimulationResultsMean,&d_lookbackSimulationResultsVariance,
       &d_lookback_VOL_0,&d_lookback_EPS_0,&d_lookback_A_0,&d_lookback_A_1,
       &d_lookback_A_2,&d_lookback_S_0,&d_lookback_MU};
    int info[]={sizeof(num_cycles),iris_r,iris_w,iris_w,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r};
    launch_iris(device,"tausworthe_lookback",active_count,LOOKBACK_TAUSWORTHE_NUM_THREADS,11,p,info);
  }


  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  {
    iris::Task task;
    check_iris(task.flush_out(d_lookbackSimulationResultsMean));
    check_iris(task.flush_out(d_lookbackSimulationResultsVariance));
    check_iris(task.submit(device,nullptr,true));
  }
  if (dump) {
    for (unsigned i = 0; i < active_count; i++)
      printf("%d %.3f %.3f\n", i, lookbackSimulationResultsMean[i], 
                              lookbackSimulationResultsVariance[i]);
  }

  if(const char* path=getenv("LOOPBACK_VALIDATION_DUMP")) {
    FILE* fp=fopen(path,"wb");if(!fp)return 1;
    fwrite(&active_count,sizeof(unsigned),1,fp);
    fwrite(lookbackSimulationResultsMean,sizeof(float),active_count,fp);
    fwrite(lookbackSimulationResultsVariance,sizeof(float),active_count,fp);
    fclose(fp);
  }
  free(lookback_VOL_0);
  free(lookback_A_0);
  free(lookback_A_1);
  free(lookback_A_2);
  free(lookback_S_0);
  free(lookback_EPS_0);
  free(lookback_MU);
  free(lookbackSimulationResultsMean);
  free(lookbackSimulationResultsVariance);
  free(tauswortheSeeds);

  const int errors=platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}
