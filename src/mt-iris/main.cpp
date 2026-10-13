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
// This sample implements Mersenne Twister random number generator
// and Cartesian Box-Muller transformation on the GPU
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#include <stdio.h>
#include "MT.h"
#include <cstdlib>
#include <cmath>
#include <iris/iris.hpp>

// comment the below line if not doing Box-Muller transformation
#define DO_BOXMULLER

// Reference CPU MT and Box-Muller transformation
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Rand, int nPerRng, unsigned int seed);
#ifdef DO_BOXMULLER
extern "C" void BoxMullerRef(float *h_Rand, int nPerRng);
#endif

#include <chrono>
using namespace std::chrono;

///////////////////////////////////////////////////////////////////////////////
//Load twister configurations
///////////////////////////////////////////////////////////////////////////////
void loadMTGPU(const char *fname,
               const unsigned int seed,
               mt_struct_stripped *h_MT,
               const size_t size)
{
  // open the file for binary read
  FILE* fd = fopen(fname, "rb");
  if (fd == NULL)
  {
    printf("Failed to open file %s\n", fname);
    exit(-1);
  }

  for (unsigned int i = 0; i < size; i++)
    fread(&h_MT[i], sizeof(mt_struct_stripped), 1, fd);
  fclose(fd);

  for(unsigned int i = 0; i < size; i++)
    h_MT[i].seed = seed;
}

///////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int numIterations = atoi(argv[1]);
  if(numIterations<=0) {fprintf(stderr,"Repeat must be positive\n");return 1;}

  size_t globalWorkSize = {MT_RNG_COUNT};  // 1D var for Total # of work items
  size_t localWorkSize = {128};            // 1D var for # of work items in the work group

  const int seed = 777;
  int nPerRng = 5860;                // # of recurrence steps, must be even if do Box-Muller transformation
  const int nRand = MT_RNG_COUNT * nPerRng;// Output size

  printf("Initialization: load MT parameters and init host buffers...\n");
  mt_struct_stripped *h_MT = (mt_struct_stripped*) malloc (
      sizeof(mt_struct_stripped) * MT_RNG_COUNT); // MT para

  const char *cDatPath = "./data/MersenneTwister.dat";
  loadMTGPU(cDatPath, seed, h_MT, MT_RNG_COUNT);

  const char *cRawPath = "./data/MersenneTwister.raw";
  initMTRef(cRawPath);

  float *h_RandGPU = (float*)malloc(sizeof(float)*nRand); // Host buffers for GPU output
  float *h_RandCPU = (float*)malloc(sizeof(float)*nRand); // Host buffers for CPU test

  printf("Allocate memory...\n");

  iris::Platform platform;
  auto check=[](int code){if(code!=IRIS_SUCCESS){fprintf(stderr,"IRIS failure%d\n",code);exit(2);}};
  check(platform.init(nullptr,nullptr,true));
  int device=-1,backend=-1,type=0;
  for(int d=0;d<platform.ndevs();++d) {
    check(iris_device_info(d,iris_backend,&backend,nullptr));
    if(backend==iris_hip){device=d;break;}
  }
  if(device<0){fprintf(stderr,"No IRIS HIP GPU\n");return 2;}
  char name[256]={};check(iris_device_info(device,iris_name,name,nullptr));check(iris_device_info(device,iris_type,&type,nullptr));
  printf("IRIS backend=%d type=%d device=%s\n",backend,type,name);
  {
    iris::DMem parameters(h_MT,sizeof(mt_struct_stripped)*MT_RNG_COUNT);
    iris::DMem output(h_RandGPU,sizeof(float)*nRand);
    printf("Call Mersenne Twister kernel... (%d iterations)\n\n",numIterations);
    high_resolution_clock::time_point t1=high_resolution_clock::now();
    for(int i=0;i<numIterations;++i) {
      iris::Task rng;
      void* params[]={&parameters,&output,&nPerRng};
      int modes[]={iris_r,iris_w,sizeof(nPerRng)};
      check(rng.kernel("mt",1,nullptr,&globalWorkSize,&localWorkSize,3,params,modes));
#ifndef DO_BOXMULLER
      if(i==numIterations-1)check(rng.flush_out(output));
#endif
      check(rng.submit(device,nullptr,true));
#ifdef DO_BOXMULLER
      iris::Task transform;
      void* p[]={&output,&nPerRng};int info[]={iris_rw,sizeof(nPerRng)};
      check(transform.kernel("boxmuller",1,nullptr,&globalWorkSize,&localWorkSize,2,p,info));
      if(i==numIterations-1)check(transform.flush_out(output));
      check(transform.submit(device,nullptr,true));
#endif
    }
    high_resolution_clock::time_point t2=high_resolution_clock::now();
    duration<double> time_span=duration_cast<duration<double>>(t2-t1);
    double gpuTime=time_span.count()/double(numIterations);
    printf("MersenneTwister, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, Workgroup = %lu\n",
           double(nRand)*1.0E-9/gpuTime,gpuTime,nRand,localWorkSize);
    printf("\nRead back results...\n");
  }
  check(platform.finalize());
  if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")) {
    std::string filename=std::string(prefix)+".bin";
    FILE* f=fopen(filename.c_str(),"wb");
    if(!f||fwrite(h_RandGPU,sizeof(float),nRand,f)!=size_t(nRand)){perror("dump");return 3;}
    fclose(f);
  }
  printf("Compute CPU reference solution...\n");
  RandomRef(h_RandCPU, nPerRng, seed);
#ifdef DO_BOXMULLER
  BoxMullerRef(h_RandCPU, nPerRng);
#endif

  printf("Compare CPU and GPU results...\n");
  double sum_delta = 0;
  double sum_ref   = 0;
  for(int i = 0; i < MT_RNG_COUNT; i++) {
    for(int j = 0; j < nPerRng; j++) {
      double rCPU = h_RandCPU[i * nPerRng + j];
      double rGPU = h_RandGPU[i + j * MT_RNG_COUNT];
      double delta = std::fabs(rCPU - rGPU);
      sum_delta += delta;
      sum_ref   += std::fabs(rCPU);
    }
  }
  double L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n\n", L1norm);

  free(h_MT);
  free(h_RandGPU);
  free(h_RandCPU);

  // finish
  printf("%s\n", (L1norm < 1e-6) ? "PASS" : "FAIL");

  return 0;
}
