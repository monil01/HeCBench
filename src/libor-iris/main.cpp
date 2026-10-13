//////////////////////////////////////////////////////////////////
//                                                              //
// This software was written by Mike Giles in 2007 based on     //
// C code written by Zhao and Glasserman at Columbia University //
//                                                              //
// It is copyright University of Oxford, and provided under     //
// the terms of the BSD3 license:                               //
// https://opensource.org/licenses/BSD-3-Clause                 //
//                                                              //
// It is provided along with an informal report on              //
// https://people.maths.ox.ac.uk/~gilesm/cuda_old.html          //
//                                                              //
// Note: this was written for CUDA 1.0 and optimised for        //
// execution on an NVIDIA 8800 GTX GPU                          //
//                                                              //
// Mike Giles, 29 April 2021                                    //
//                                                              //
//////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "iris_support.h"

// parameters for device execution

#define BLOCK_SIZE 64
#define GRID_SIZE 1500

// parameters for LIBOR calculation

#define NN 80
#define NMAT 40
#define L2_SIZE 3280 //NN*(NMAT+1)
#define NOPT 15
#define NPATH 96000

// Monte Carlo LIBOR path calculation

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  if(repeat<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();
    
  // 'h_' prefix - CPU (host) memory space

  float  *h_v, *h_Lb, h_lambda[NN], h_delta=0.25f;
  int     h_N=NN, h_Nmat=NMAT, h_Nopt=NOPT, i;
  int     h_maturities[] = {4,4,4,8,8,8,20,20,20,28,28,28,40,40,40};
  float   h_swaprates[]  = {.045f,.05f,.055f,.045f,.05f,.055f,.045f,.05f,
                            .055f,.045f,.05f,.055f,.045f,.05f,.055f };
  double  v, Lb; 
  bool    ok = true;

  // 'd_' prefix - GPU (device) memory space


  for (i=0; i<NN; i++) h_lambda[i] = 0.2f;

  h_v      = (float *)malloc(sizeof(float)*NPATH);
  h_Lb     = (float *)malloc(sizeof(float)*NPATH);

  iris::DMem d_v(h_v,sizeof(float)*NPATH), d_Lb(h_Lb,sizeof(float)*NPATH);
  iris::DMem d_maturities(h_maturities,sizeof(h_maturities));
  iris::DMem d_swaprates(h_swaprates,sizeof(h_swaprates));
  iris::DMem d_lambda(h_lambda,sizeof(h_lambda));
  size_t lws=BLOCK_SIZE, gws=BLOCK_SIZE*GRID_SIZE;
  auto launch = [&](bool greeks) {
    iris::Task task;
    if(greeks) {
      void* args[]={&d_v,&d_Lb,&d_lambda,&d_maturities,&d_swaprates,&h_delta,&h_Nmat,&h_N,&h_Nopt};
      int info[]={iris_w,iris_w,iris_r,iris_r,iris_r,sizeof(h_delta),sizeof(h_Nmat),sizeof(h_N),sizeof(h_Nopt)};
      iris_check(task.kernel("Pathcalc_Portfolio_KernelGPU",1,nullptr,&gws,&lws,9,args,info),"Greeks kernel");
    } else {
      void* args[]={&d_v,&d_lambda,&d_maturities,&d_swaprates,&h_delta,&h_Nmat,&h_N,&h_Nopt};
      int info[]={iris_w,iris_r,iris_r,iris_r,sizeof(h_delta),sizeof(h_Nmat),sizeof(h_N),sizeof(h_Nopt)};
      iris_check(task.kernel("Pathcalc_Portfolio_KernelGPU2",1,nullptr,&gws,&lws,8,args,info),"Portfolio kernel");
    }
    iris_check(task.submit(device,nullptr,true),"HIP task");
  };
  auto flush_values = [&](bool greeks) {
    iris::Task task;
    iris_check(task.flush_out(d_v),"flush values");
    if(greeks) iris_check(task.flush_out(d_Lb),"flush sensitivities");
    iris_check(task.submit(device,nullptr,true),"HIP flush");
  };

  // Launch the device computation threads
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    launch(false);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

  // Read back GPU results and compute average
  flush_values(false);

  v = 0.0;
  for (i=0; i<NPATH; i++) v += h_v[i];
  v = v / NPATH;

  if (!std::isfinite(v) || fabs(v - 224.323) > 1e-3) {
    ok = false;
    printf("Expected: 224.323 Actual %15.3f\n", v);
  }

  // Execute GPU kernel -- Greeks

  // Launch the device computation threads
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    launch(true);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (s)\n", (time * 1e-9f) / repeat);

  // Read back GPU results and compute average
  flush_values(true);

  v = 0.0;
  for (i=0; i<NPATH; i++) v += h_v[i];
  v = v / NPATH;

  Lb = 0.0;
  for (i=0; i<NPATH; i++) Lb += h_Lb[i];
  Lb = Lb / NPATH;

  if (!std::isfinite(v) || fabs(v - 224.323) > 1e-3) {
    ok = false;
    printf("Expected: 224.323 Actual %15.3f\n", v);
  }
  if (!std::isfinite(Lb) || fabs(Lb - 21.348) > 1e-3) {
    ok = false;
    printf("Expected:  21.348 Actual %15.3f\n", Lb);
  }

       
  free(h_v);
  free(h_Lb);

  printf("%s\n", ok ? "PASS" : "FAIL");

  iris_check(platform.finalize(),"finalize");
  return ok ? 0 : 1;
}
