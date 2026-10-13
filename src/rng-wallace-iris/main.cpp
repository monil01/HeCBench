#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iris/iris.hpp>
#include <stdexcept>

#include "rand_helpers.h"
#include "constants.h"

#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // host buffers
  float *hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  float *Pool_ref = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);

  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++) {
    float x = RandN();
    Pool_ref[i] = hostPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
    rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);
  float* randomNumbers_ref = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

  // device buffers
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  iris::DMem devPool(hostPool,4*WALLACE_TOTAL_POOL_SIZE);
  iris::DMem devicerngChi2Corrections(rngChi2Corrections,4*WALLACE_CHI2_COUNT);
  iris::DMem device_randomNumbers(randomNumbers,4*WALLACE_OUTPUT_SIZE);

  size_t global[]={size_t(WALLACE_NUM_BLOCKS)*WALLACE_NUM_THREADS},local[]={WALLACE_NUM_THREADS};
  auto launch=[&](const unsigned seed){
    iris::Task task;void* params[]={const_cast<unsigned*>(&seed),&devPool,&device_randomNumbers,&devicerngChi2Corrections};
    int info[]={sizeof(seed),iris_rw,iris_w,iris_r};
    if(task.kernel("rng_wallace",1,nullptr,global,local,4,params,info)||task.submit(device,nullptr,1)) throw std::runtime_error("Wallace task failed");
  };
  auto flush=[&]() {
    iris::Task task;task.flush_out(devPool);task.flush_out(device_randomNumbers);
    if(task.submit(device,nullptr,1)) throw std::runtime_error("Wallace flush failed");
  };
  const unsigned seed = 1;

  // warmup and quick validation
  for (int i = 0; i < 30; i++) {
    launch(seed);
  }

  for (int i = 0; i < 30; i++) {
    reference(seed, Pool_ref, randomNumbers_ref, rngChi2Corrections, WALLACE_NUM_BLOCKS);
  }

  flush();

  bool ok = true;
  for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) {
    if (fabsf(randomNumbers_ref[n] - randomNumbers[n]) > 1e-3f) {
      printf("randNumbers mismatch at index %d: %f %f\n", n, randomNumbers_ref[n], randomNumbers[n]);
      ok = false;
      break;
    }
  }
  for (unsigned int n = 0; n < WALLACE_TOTAL_POOL_SIZE; n++) {
    if (fabsf(Pool_ref[n] - hostPool[n]) > 1e-3f) {
      printf("Pool mismatch at index %d: %f %f\n", n, Pool_ref[n], hostPool[n]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if(!ok) return 1;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    launch(seed);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", time * 1e-3f / repeat);

  flush();
  if(const char* path=getenv("HECBENCH_WALLACE_OUTPUT")) {
    FILE* fp=fopen(path,"wb"); if(!fp) return 2;
    fwrite(randomNumbers,sizeof(float),WALLACE_OUTPUT_SIZE,fp);
    fwrite(hostPool,sizeof(float),WALLACE_TOTAL_POOL_SIZE,fp);
    fclose(fp);
  }
  platform.finalize();
  free(rngChi2Corrections);
  free(randomNumbers);
  free(randomNumbers_ref);
  free(hostPool);
  free(Pool_ref);
  return 0;
}
