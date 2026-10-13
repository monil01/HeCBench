/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 * 
 * Tridiagonal solvers.
 * Host code for parallel pcr reduction (PCR).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _PCR_SMALL_SYSTEMS_
#define _PCR_SMALL_SYSTEMS_

#include "task_launch.h"
#include "tridiagonal.h"

const char *pcrKernelNames[] = { 
  "pcr_small_systems_kernel",    // original version
  "pcr_branch_free_kernel",      // optimized branch-free version
};  

double pcr_small_systems(float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, int id = 0)
{
  shrLog(" %s\n", pcrKernelNames[id]);

  const unsigned int mem_size = num_systems * system_size * sizeof(float);
  std::vector<float> backing_a(a,a+num_systems*system_size);
  iris::DMem mem_a(backing_a.data(),mem_size);iris::DMem* a_d=&mem_a;
  std::vector<float> backing_b(b,b+num_systems*system_size);
  iris::DMem mem_b(backing_b.data(),mem_size);iris::DMem* b_d=&mem_b;
  std::vector<float> backing_c(c,c+num_systems*system_size);
  iris::DMem mem_c(backing_c.data(),mem_size);iris::DMem* c_d=&mem_c;
  std::vector<float> backing_d(d,d+num_systems*system_size);
  iris::DMem mem_d(backing_d.data(),mem_size);iris::DMem* d_d=&mem_d;
  std::vector<float> backing_x(num_systems*system_size);
  iris::DMem mem_x(backing_x.data(),mem_size);iris::DMem* x_d=&mem_x;

  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;
  int iterations = log2(system_size/2);

  // set execution parameters
  szLocalWorkSize = system_size;
  szGlobalWorkSize = num_systems; 

  dim3 gws (szGlobalWorkSize);
  dim3 lws (szLocalWorkSize);

  // warm up
  if (id == 0)
    launch_task("pcr_small_systems_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,&system_size,&num_systems,&iterations},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(iterations)});
  else
    launch_task("pcr_branch_free_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,&system_size,&num_systems,&iterations},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(iterations)});


  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (id == 0)
      launch_task("pcr_small_systems_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,&system_size,&num_systems,&iterations},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(iterations)});
    else
      launch_task("pcr_branch_free_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,&system_size,&num_systems,&iterations},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(iterations)});
  }

  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  // write-back to the array x
  iris::Task done;done.flush_out(*x_d);if(done.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(2);
  if(x_d->fetch(x)!=IRIS_SUCCESS)exit(2); 





  return time;
}
#endif
