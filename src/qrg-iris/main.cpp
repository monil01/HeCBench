/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

///////////////////////////////////////////////////////////////////////////////
// This sample implements Niederreiter quasirandom number generator
// and Moro's Inverse Cumulative Normal Distribution generator
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}

#include "qrg.h"

// forward declarations
void initQuasirandomGenerator(unsigned int *table);
double getQuasirandomValue63(INT64 i, int dim);
double MoroInvCNDcpu(unsigned int x);

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size)
{
  int r = global_size % group_size;
  if(r == 0)
  {
    return global_size;
  } else
  {
    return global_size + group_size - r;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Moro's Inverse Cumulative Normal Distribution function approximation
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  if (argc != 2 && argc != 3) {
    printf("Usage: %s <repeat> [bounded size]\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const unsigned int N=argc==3 ? strtoul(argv[2],nullptr,10) : 1048576;
  if(N==0 || N>1048576)return 1;

  unsigned int dim, pos;
  double delta, ref, sumDelta, sumRef, L1norm;
  unsigned int tableCPU[QRNG_DIMENSIONS*QRNG_RESOLUTION];
  bool bPassFlag = false;

  float* h_OutputGPU = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

  printf("Initializing QRNG tables...\n");
  initQuasirandomGenerator(tableCPU);

  iris::Platform platform;check(platform.init(&argc,&argv,true));int device=hip_device();
  iris::DMem d_Output(h_OutputGPU,sizeof(float)*QRNG_DIMENSIONS*N);
  iris::DMem d_Table(tableCPU,sizeof(unsigned int)*QRNG_DIMENSIONS*QRNG_RESOLUTION);
  printf(">>>Launch QuasirandomGenerator kernel...\n\n");

  size_t szWorkgroup = 64 * (256 / QRNG_DIMENSIONS)/64;
  size_t globalWorkSize[2] = {shrRoundUp(szWorkgroup, 128*128), QRNG_DIMENSIONS};
  size_t localWorkSize[2] = {szWorkgroup, QRNG_DIMENSIONS};


  // seed is fixed at zero
  const unsigned int seed = 0;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
  {
    void* args[]={&d_Output,&d_Table,arg(seed),arg(N)};
    int info[]={iris_w,iris_r,sizeof(seed),sizeof(N)};
    iris::Task task;check(task.kernel("qrng",2,nullptr,globalWorkSize,localWorkSize,4,args,info));
    check(task.submit(device,nullptr,true));
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (qrng): %f (us)\n", (time * 1e-3f) / repeat);

  printf("\nRead back results...\n");
  {iris::Task flush;check(flush.flush_out(d_Output));check(flush.submit(device,nullptr,true));}

  printf("Comparing to the CPU results...\n\n");
  sumDelta = 0;
  sumRef   = 0;
  for(dim = 0; dim < QRNG_DIMENSIONS; dim++)
  {
    for(pos = 0; pos < N; pos++)
    {
      ref       = getQuasirandomValue63(pos, dim);
      delta     = (double)h_OutputGPU[dim * N  + pos] - ref;
      sumDelta += fabs(delta);
      sumRef   += fabs(ref);
    }
  }
  L1norm = sumDelta / sumRef;
  printf("  L1 norm: %E\n", L1norm);
  printf("  ckQuasirandomGenerator deviations %s Allowable Tolerance\n\n\n", (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
  bPassFlag = (L1norm < 1e-6);

  printf(">>>Launch InverseCND kernel...\n\n");

  // reuse variables for work-group sizes
  szWorkgroup = 128;
  globalWorkSize[0] = shrRoundUp(szWorkgroup, 128*128);
  localWorkSize[0] = szWorkgroup;



  const unsigned int pathN = QRNG_DIMENSIONS * N;
  const unsigned int distance = ((unsigned int)-1) / (pathN  + 1);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
  {
    void* args[]={&d_Output,arg(pathN),arg(distance)};int info[]={iris_w,sizeof(pathN),sizeof(distance)};
    iris::Task task;check(task.kernel("icnd",1,nullptr,globalWorkSize,localWorkSize,3,args,info));
    check(task.submit(device,nullptr,true));
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (icnd): %f (us)\n", (time * 1e-3f) / repeat);

  printf("\nRead back results...\n");
  {iris::Task flush;check(flush.flush_out(d_Output));check(flush.submit(device,nullptr,true));}

  printf("Comparing to the CPU results...\n\n");
  sumDelta = 0;
  sumRef   = 0;
  for(pos = 0; pos < QRNG_DIMENSIONS * N; pos++){
    unsigned int d = (pos + 1) * distance;
    ref       = MoroInvCNDcpu(d);
    delta     = (double)h_OutputGPU[pos] - ref;
    sumDelta += fabs(delta);
    sumRef   += fabs(ref);
  }
  L1norm = sumDelta / sumRef;
  printf("  L1 norm: %E\n", L1norm);
  printf("  ckInverseCNDGPU deviations %s Allowable Tolerance\n\n\n", (L1norm < 1e-6) ? "WITHIN" : "ABOVE");
  bPassFlag &= (L1norm < 1e-6);

  if (bPassFlag)
    printf("PASS\n");
  else
    printf("FAIL\n");

  free(h_OutputGPU);
  check(platform.finalize());
  return bPassFlag ? 0 : 1;
}
