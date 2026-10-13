// Modified: CUDA host allocations/launches replaced with IRIS DMem/tasks.
#include <chrono>
#include <iostream>
#include "iris_support.h"
#include "ThomasMatrix.hpp"
#include "utils.hpp"

// CPU kernel
void solve_seq(const double* l, const double* d,
               double* u, double* rhs,
               const int n, const int N) 
{
  int first,last;
  for (int j = 0; j < N; ++j)
  {
    first = j*n;
    last = first + n - 1;

    u[first] /= d[first];
    rhs[first] /= d[first];

    for (int i = first+1; i < last; i++) {
      u[i] /= d[i] - l[i]*u[i-1];
      rhs[i] = (rhs[i] - l[i]*rhs[i-1]) / (d[i] - l[i]*u[i-1]);
    }

    rhs[last] = (rhs[last] - l[last]*rhs[last-1]) / (d[last] - l[last]*u[last-1]);

    for (int i = last-1; i >= first; i--) {
      rhs[i] -= u[i]*rhs[i+1];
    }
  }
}

int main(int argc, char *argv[])
{
  if(argc != 5) {
    std::cout << "Usage: %s [system size] [#systems] [thread block size] [repeat]" << std::endl;
    return -1;
  }

  const int M = std::stoi(argv[1]);
  const int N = std::stoi(argv[2]);
  const int BlockSize = std::stoi(argv[3]);  // GPU thread block size
  const int repeat = std::stoi(argv[4]);
  if(M<2 || N<1 || BlockSize<1 || BlockSize>1024 || repeat<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();

  const size_t matrix_size = (size_t)M * N;
  const size_t matrix_size_bytes = matrix_size * sizeof(double);

  //Loading a synthetic tridiagonal matrix into our structure
  ThomasMatrix params = loadThomasMatrixSyn(M);

  double* u_seq = (double*) malloc(matrix_size_bytes);
  double* u_Thomas_host =  (double*) malloc(matrix_size_bytes);
  double* u_input = (double*) malloc(matrix_size_bytes);

  double* d_seq = (double*) malloc(matrix_size_bytes);
  double* d_Thomas_host =  (double*) malloc(matrix_size_bytes);
  double* d_input = (double*) malloc(matrix_size_bytes);

  double* l_seq = (double*) malloc(matrix_size_bytes);
  double* l_Thomas_host =  (double*) malloc(matrix_size_bytes);
  double* l_input = (double*) malloc(matrix_size_bytes);

  double* rhs_seq = (double*) malloc(matrix_size_bytes);
  double* rhs_Thomas_host = (double*) malloc(matrix_size_bytes);
  double* rhs_input = (double*) malloc(matrix_size_bytes);

  double* rhs_seq_output = (double*) malloc(matrix_size_bytes);
  double* rhs_seq_interleave = (double*) malloc(matrix_size_bytes);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      u_seq[(i * M) + j] = params.a[j];
      u_input[(i * M) + j] = params.a[j];

      d_seq[(i * M) + j] = params.d[j];
      d_input[(i * M) + j] = params.d[j];

      l_seq[(i * M) + j] = params.b[j];
      l_input[(i * M) + j] = params.b[j];

      rhs_seq[(i * M) + j] = params.rhs[j];
      rhs_input[(i * M) + j] = params.rhs[j];
    }
  }

  auto start = std::chrono::steady_clock::now();

  // Sequantial CPU Execution for correct error check
  for (int n = 0; n < repeat; n++) {
    solve_seq( l_seq, d_seq, u_seq, rhs_seq, M, N );
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average serial execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  for (size_t i = 0; i < matrix_size; ++i) {
    rhs_seq_output[i] = rhs_seq[i];
  }

  // initialize again because u_seq and rhs_seq are modified by solve_seq
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < M; ++j)
    {
      u_seq[(i * M) + j] = params.a[j];
      u_input[(i * M) + j] = params.a[j];

      d_seq[(i * M) + j] = params.d[j];
      d_input[(i * M) + j] = params.d[j];

      l_seq[(i * M) + j] = params.b[j];
      l_input[(i * M) + j] = params.b[j];

      rhs_seq[(i * M) + j] = params.rhs[j];
      rhs_input[(i * M) + j] = params.rhs[j];

    }
  }

  // transpose the inputs for sequential accesses on a GPU 
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      u_Thomas_host[i*N+j] = u_input[j*M+i];
      l_Thomas_host[i*N+j] = l_input[j*M+i];
      d_Thomas_host[i*N+j] = d_input[j*M+i];
      rhs_Thomas_host[i*N+j] = rhs_input[j*M+i];
      rhs_seq_interleave[i*N+j] = rhs_seq_output[j*M+i];
    }
  }
 
  // Run GPU kernel

  iris::DMem u_device(u_Thomas_host,matrix_size_bytes);
  iris::DMem l_device(l_Thomas_host,matrix_size_bytes);
  iris::DMem d_device(d_Thomas_host,matrix_size_bytes);
  iris::DMem rhs_device(rhs_Thomas_host,matrix_size_bytes);
  size_t lws=BlockSize, gws=(N/BlockSize+1)*lws;
  start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* args[]={&l_device,&d_device,&u_device,&rhs_device,(void*)&M,(void*)&N};
    int info[]={iris_r,iris_r,iris_rw,iris_rw,sizeof(M),sizeof(N)};
    iris_check(task.kernel("cuThomasBatch",1,nullptr,&gws,&lws,6,args,info),"kernel");
    iris_check(task.submit(device,nullptr,true),"HIP task");
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  iris::Task flush;
  iris_check(flush.flush_out(rhs_device),"flush rhs");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");

  // verify
  calcError(rhs_seq_interleave, rhs_Thomas_host, matrix_size);
  bool finite=true;
  for(size_t i=0;i<matrix_size;++i) finite &= std::isfinite(rhs_Thomas_host[i]) && std::isfinite(rhs_seq_interleave[i]);
  if(!finite) fprintf(stderr,"Nonfinite solution\n");

  free(u_seq);  
  free(u_Thomas_host);
  free(u_input);

  free(d_seq);  
  free(d_Thomas_host);
  free(d_input);

  free(l_seq);  
  free(l_Thomas_host);
  free(l_input);

  free(rhs_seq);  
  free(rhs_Thomas_host);
  free(rhs_input);

  free(rhs_seq_output);
  free(rhs_seq_interleave);


  iris_check(platform.finalize(),"finalize");
  return finite ? 0 : 1;
}
