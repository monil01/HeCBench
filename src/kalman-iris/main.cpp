/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <#series> <#observations> <forcast steps> <repeat>\n", argv[0]);
    return 1;
  }

  const int nseries = atoi(argv[1]);
  const int nobs = atoi(argv[2]);
  const int fc_steps = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int rd = 8;
  const int rd2 = rd * rd;
  const int batch_size = nseries;
  const int rd2_size = nseries * rd2 * sizeof(double);
  const int rd_size = nseries * rd * sizeof(double);
  const int nobs_size = nseries * nobs * sizeof(double);
  const int ns_size = nseries * sizeof(double);
  const int fc_size = fc_steps * nseries * sizeof(double);

  int i;
  srand(123);
  double *RQR = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    RQR[i] = (double)rand() / (double)RAND_MAX;


  double *T = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    T[i] = 1.0;


  double *P = (double*) malloc (rd2_size);
  for (i = 0; i < rd2 * nseries; i++)
    P[i] = (double)rand() / (double)RAND_MAX;


  double *Z = (double*) malloc (rd_size);
  for (i = 0; i < rd * nseries; i++)
    Z[i] = (double)rand() / (double)RAND_MAX;


  double *alpha = (double*) malloc (rd_size);
  for (i = 0; i < rd * nseries; i++)
    alpha[i] = (double)rand() / (double)RAND_MAX;


  double *ys = (double*) malloc (nobs_size);
  for (i = 0; i < nobs * nseries; i++)
    ys[i] = (double)rand() / (double)RAND_MAX;


  double *mu = (double*) malloc (ns_size);
  for (i = 0; i < nseries; i++)
    mu[i] = (double)rand() / (double)RAND_MAX;


  double *vs = (double*) malloc (nobs_size);

  double *Fs = (double*) malloc (nobs_size);

  double *sum_logFs = (double*) malloc (ns_size);

  double *fc = (double*) malloc (fc_size);

  double *F_fc = (double*) malloc (fc_size);

  iris::Platform platform;check(platform.init(&argc,&argv,true));int device=hip_device();
  iris::DMem d_RQR(RQR,rd2_size);
  iris::DMem d_T(T,rd2_size);
  iris::DMem d_P(P,rd2_size);
  iris::DMem d_Z(Z,rd_size);
  iris::DMem d_alpha(alpha,rd_size);
  iris::DMem d_ys(ys,nobs_size);
  iris::DMem d_mu(mu,ns_size);
  iris::DMem d_vs(vs,nobs_size);
  iris::DMem d_Fs(Fs,nobs_size);
  iris::DMem d_sum_logFs(sum_logFs,ns_size);
  iris::DMem d_fc(fc,fc_size);
  iris::DMem d_F_fc(F_fc,fc_size);
  bool intercept=true, conf_int=true;
  size_t gws=size_t((nseries+255)/256)*256,lws=256;

  for (int n_diff = 0; n_diff < rd; n_diff++) {

    auto start = std::chrono::steady_clock::now();

    for(i=0;i<repeat;++i) {
      void* args[]={&d_ys,arg(nobs),&d_T,&d_Z,&d_RQR,&d_P,&d_alpha,&intercept,&d_mu,arg(batch_size),
                    &d_vs,&d_Fs,&d_sum_logFs,&n_diff,arg(fc_steps),&d_fc,&conf_int,&d_F_fc};
      int info[]={iris_r,sizeof(nobs),iris_r,iris_r,iris_r,iris_r,iris_r,sizeof(intercept),iris_r,sizeof(batch_size),
                  iris_w,iris_w,iris_w,sizeof(n_diff),sizeof(fc_steps),iris_w,sizeof(conf_int),iris_w};
      iris::Task task;check(task.kernel("kalman8",1,nullptr,&gws,&lws,18,args,info));
      check(task.submit(device,nullptr,true));
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (n_diff = %d): %f (s)\n", n_diff, (time * 1e-9f) / repeat);
    iris::Task flush;check(flush.flush_out(d_F_fc));check(flush.submit(device,nullptr,true));
      reference<rd>(
          nseries,
          nobs,
          ys,
          T,
          Z,
          RQR,
          P,
          alpha,
          mu,
          F_fc, // device
          true, // intercept,
          n_diff,
          fc_steps,
          true); // forcast
  }

  free(fc);
  free(F_fc);
  free(sum_logFs);
  free(mu);
  free(Fs);
  free(vs);
  free(ys);
  free(alpha);
  free(Z);
  free(P);
  free(T);
  free(RQR);
  check(platform.finalize());
  return 0;
}
