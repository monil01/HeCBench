#include <vector>
#include <iostream>
#include <chrono>
#include "dmem_runtime.hpp"
#include "util.h"  // graph

#define DIAMETER_SAMPLES 512
#include "dispatch.h"

#define checkCudaErrors(err) iris_check(err)

std::vector<float> bc_gpu(
  graph g,
  int max_threads_per_block,
  int number_of_SMs,
  program_options op,
  const std::set<int> &source_vertices)
{
  float *bc_gpu = new float[g.n];
  int next_source = number_of_SMs; 

  float *bc_d, *delta_d;
  int *d_d, *R_d, *C_d, *F_d, *Q_d, *Q2_d, *S_d, *endpoints_d, *next_source_d, *source_vertices_d;
  unsigned long long *sigma_d;
  size_t pitch_d, pitch_sigma, pitch_delta, pitch_Q, pitch_Q2, pitch_S, pitch_endpoints;
  int *jia_d, *diameters_d;

  dim3 dimGrid (number_of_SMs, 1, 1);
  dim3 dimBlock (max_threads_per_block, 1, 1); 

  //Allocate and transfer data to the GPU
  checkCudaErrors(iris_alloc((void**)&bc_d,sizeof(float)*g.n));
  checkCudaErrors(iris_alloc((void**)&R_d,sizeof(int)*(g.n+1)));
  checkCudaErrors(iris_alloc((void**)&C_d,sizeof(int)*(2*g.m)));
  checkCudaErrors(iris_alloc((void**)&F_d,sizeof(int)*(2*g.m)));

  checkCudaErrors(iris_alloc_pitch((void**)&d_d,&pitch_d,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(iris_alloc_pitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,dimGrid.x));
  checkCudaErrors(iris_alloc_pitch((void**)&delta_d,&pitch_delta,sizeof(float)*g.n,dimGrid.x));
  //Making Queues/Stack of size O(n) since we won't duplicate
  checkCudaErrors(iris_alloc_pitch((void**)&Q_d,&pitch_Q,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(iris_alloc_pitch((void**)&Q2_d,&pitch_Q2,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(iris_alloc_pitch((void**)&S_d,&pitch_S,sizeof(int)*g.n,dimGrid.x));
  checkCudaErrors(iris_alloc_pitch((void**)&endpoints_d,&pitch_endpoints,sizeof(int)*(g.n+1),dimGrid.x));

  checkCudaErrors(iris_alloc((void**)&next_source_d,sizeof(int)));

  // source_vertices of type "std::set" has no data() method
  std::vector<int> source_vertices_h(source_vertices.size());
  std::copy(source_vertices.begin(),source_vertices.end(),source_vertices_h.begin());
  checkCudaErrors(iris_alloc((void**)&source_vertices_d, sizeof(int) * source_vertices.size()));
  
  if(op.approx)
  {
    checkCudaErrors(iris_copy(source_vertices_d, source_vertices_h.data(),
                    sizeof(int) * source_vertices.size(), iris_h2d));
  }

  checkCudaErrors(iris_alloc((void**)&jia_d,sizeof(int)));
  checkCudaErrors(iris_alloc((void**)&diameters_d,sizeof(int)*DIAMETER_SAMPLES));
  checkCudaErrors(iris_fill(jia_d,0,sizeof(int)));
  checkCudaErrors(iris_fill(diameters_d,0,sizeof(int)*DIAMETER_SAMPLES));

  checkCudaErrors(iris_copy(R_d,g.R,sizeof(int)*(g.n+1),iris_h2d));
  checkCudaErrors(iris_copy(C_d,g.C,sizeof(int)*(2*g.m),iris_h2d));
  checkCudaErrors(iris_copy(F_d,g.F,sizeof(int)*(2*g.m),iris_h2d));
  checkCudaErrors(iris_fill(bc_d,0,sizeof(float)*g.n));
  checkCudaErrors(iris_copy(next_source_d,&next_source,sizeof(int),iris_h2d));

  int end;
  bool approx;
  if(op.approx)
  { 
    end = op.k;
    approx = true;
  } else {
    end = g.n;
    approx = false;
  }

  iris_sync();
  auto start = std::chrono::steady_clock::now();
  
  dispatch_bc_kernel( dim3(dimGrid), dim3(dimBlock), 
      bc_d,
      R_d,
      C_d,
      F_d,
      g.n,
      g.m,
      d_d,
      sigma_d,
      delta_d,
      Q_d,
      Q2_d,
      S_d,
      endpoints_d,
      next_source_d,
      pitch_d,
      pitch_sigma,
      pitch_delta,
      pitch_Q,
      pitch_Q2,
      pitch_S,
      pitch_endpoints,
      0,
      end,
      jia_d,
      diameters_d,
      source_vertices_d,
      approx);

  iris_sync();
  auto stop = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  std::cout << "Kernel execution time " << time * 1e-9f << " (s)\n";

  // GPU result
  checkCudaErrors(iris_copy(bc_gpu,bc_d,sizeof(float)*g.n,iris_d2h));

  checkCudaErrors(iris_free(bc_d));
  checkCudaErrors(iris_free(R_d));
  checkCudaErrors(iris_free(C_d));
  checkCudaErrors(iris_free(F_d));
  checkCudaErrors(iris_free(d_d));
  checkCudaErrors(iris_free(sigma_d));
  checkCudaErrors(iris_free(delta_d));
  checkCudaErrors(iris_free(Q_d));
  checkCudaErrors(iris_free(Q2_d));
  checkCudaErrors(iris_free(S_d));
  checkCudaErrors(iris_free(endpoints_d));
  checkCudaErrors(iris_free(next_source_d));
  checkCudaErrors(iris_free(jia_d));
  checkCudaErrors(iris_free(diameters_d));
  checkCudaErrors(iris_free(source_vertices_d));

  //Copy GPU result to a vector
  std::vector<float> bc_gpu_v(bc_gpu,bc_gpu+g.n);

  for(int i=0; i<g.n; i++)
  {
    bc_gpu_v[i] /= 2.0f; //we don't want to double count the unweighted edges
  }

  delete[] bc_gpu;
  return bc_gpu_v;
}

// query the properties of a single device for simplicity
void query_device(int &max_threads_per_block, int &number_of_SMs, program_options op)
{
  op.device = 0;
  checkCudaErrors(hipSetDevice(op.device));
  hipDeviceProp_t prop;
  checkCudaErrors(hipGetDeviceProperties(&prop, op.device));

  std::cout << "Chosen Device: " << prop.name << std::endl;
  std::cout << "Number of Multiprocessors: " << prop.multiProcessorCount << std::endl;
  std::cout << "Size of Global Memory: " << prop.totalGlobalMem/(float)(1024*1024*1024)
            << " GB" << std::endl << std::endl;

  max_threads_per_block = prop.maxThreadsPerBlock;
  number_of_SMs = prop.multiProcessorCount;
}
