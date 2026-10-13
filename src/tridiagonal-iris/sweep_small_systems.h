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

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 * 
 * Tridiagonal solvers.
 * Host code for sweep solver (one-system-per-thread).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _SWEEP_SMALL_SYSTEMS_
#define _SWEEP_SMALL_SYSTEMS_

#include "task_launch.h"
#include <algorithm>   // std::swap
#include "tridiagonal.h"


const char *sweepKernelNames[] = { 
  "sweep_small_systems_local_kernel",      // use local memory for temp array
  "sweep_small_systems_global_kernel",    // use global memory for temp array
  "sweep_small_systems_global_vec4_kernel",  // use global memory abd solve 4 systems per thread
  "transpose",              // data reordering
};  

double runReorderKernel(iris::DMem* d_a, iris::DMem* d_t, int width, int height)
{
  size_t szGlobalWorkSize[2];
  size_t szLocalWorkSize[2];

  // set transpose kernel execution parameters
  szLocalWorkSize[0] = TRANSPOSE_BLOCK_DIM;
  szLocalWorkSize[1] = TRANSPOSE_BLOCK_DIM;
  szGlobalWorkSize[0] = shrRoundUp(TRANSPOSE_BLOCK_DIM, width) / TRANSPOSE_BLOCK_DIM;
  szGlobalWorkSize[1] = shrRoundUp(TRANSPOSE_BLOCK_DIM, height) / TRANSPOSE_BLOCK_DIM;
  dim3 gws (szGlobalWorkSize[0], szGlobalWorkSize[1]);
  dim3 lws (szLocalWorkSize[0], szLocalWorkSize[1]);

  launch_task("transpose",gws,lws,{d_t,d_a,&width,&height},{iris_rw,iris_rw,sizeof(width),sizeof(height)});


  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    launch_task("transpose",gws,lws,{d_t,d_a,&width,&height},{iris_rw,iris_rw,sizeof(width),sizeof(height)});
  }


  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double runSweepKernel(
    iris::DMem* a_d, 
    iris::DMem* b_d, 
    iris::DMem* c_d, 
    iris::DMem* d_d, 
    iris::DMem* x_d, 
    iris::DMem* t_d, 
    iris::DMem* w_d, 
    int system_size, 
    int num_systems,
    bool reorder)
{
  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;

  // set main kernel execution parameters
  if (useVec4) szLocalWorkSize = SWEEP_BLOCK_SIZE / 4;
  else szLocalWorkSize = SWEEP_BLOCK_SIZE;
  szGlobalWorkSize = shrRoundUp(SWEEP_BLOCK_SIZE, num_systems) / szLocalWorkSize;

  dim3 gws (szGlobalWorkSize);
  dim3 lws (szLocalWorkSize);

  // warm up
  if (useLmem) 
    launch_task("sweep_small_systems_local_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,&system_size,&num_systems,&reorder},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(reorder)});
  else if (useVec4) 
    launch_task("sweep_small_systems_global_vec4_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,w_d,&system_size,&num_systems,&reorder},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(reorder)});
  else 
    launch_task("sweep_small_systems_global_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,w_d,&system_size,&num_systems,&reorder},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(reorder)});


  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (useLmem) 
      launch_task("sweep_small_systems_local_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,&system_size,&num_systems,&reorder},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(reorder)});
    else if (useVec4) 
      launch_task("sweep_small_systems_global_vec4_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,w_d,&system_size,&num_systems,&reorder},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(reorder)});
    else 
      launch_task("sweep_small_systems_global_kernel",gws,lws,{a_d,b_d,c_d,d_d,x_d,w_d,&system_size,&num_systems,&reorder},{iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(system_size),sizeof(num_systems),sizeof(reorder)});
  }

  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double sweep_small_systems(float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, bool reorder = false)
{
  if (reorder) shrLog("sweep_data_reorder_kernel\n"); 
  if (useLmem) shrLog("%s\n", sweepKernelNames[0]); 
  else if (useVec4) shrLog("%s\n", sweepKernelNames[2]); 
  else shrLog("%s\n", sweepKernelNames[1]); 

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
  std::vector<float> backing_t(num_systems*system_size);
  iris::DMem mem_t(backing_t.data(),mem_size);iris::DMem* t_d=&mem_t;
  std::vector<float> backing_w(num_systems*system_size);
  iris::DMem mem_w(backing_w.data(),mem_size);iris::DMem* w_d=&mem_w;

  int workSize = num_systems;

  double reorder_time = 0.0;
  double solver_time = 0.0;

  if (reorder)
  {
    // transpose input data
    reorder_time += runReorderKernel(a_d, t_d, system_size, workSize);
    std::swap(a_d, t_d);

    reorder_time += runReorderKernel(b_d, t_d, system_size, workSize);
    std::swap(b_d, t_d);

    reorder_time += runReorderKernel(c_d, t_d, system_size, workSize);
    std::swap(c_d, t_d);

    reorder_time += runReorderKernel(d_d, t_d, system_size, workSize);
    std::swap(d_d, t_d);
  }

  // run solver
  solver_time = runSweepKernel(a_d, b_d, c_d, d_d, 
      x_d, t_d, w_d, system_size, workSize, reorder);

  if (reorder)
  {
    // transpose result back
    reorder_time += runReorderKernel(x_d, t_d, workSize, system_size);
    std::swap(x_d, t_d);
  }
  
  // copy result from device to host 
  iris::Task done;done.flush_out(*x_d);if(done.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(2);
  if(x_d->fetch(x)!=IRIS_SUCCESS)exit(2); 








  return solver_time + reorder_time;
}

#endif
