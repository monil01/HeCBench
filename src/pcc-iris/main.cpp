#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include "device.h"
#include "iris_support.hpp"
#include <vector>
#include <algorithm>
#include <cstdlib>
int device;
size_t memory_budget() { return getenv("PCC_VALIDATION_MEMORY_BYTES") ? strtoull(getenv("PCC_VALIDATION_MEMORY_BYTES"),nullptr,10) : FREE_MEMORY; }


int CorMat_singlePass(float* , float * , int , int);
int CorMat_multiPass(float* , float * , int , int);

size_t remaining_B(int N, size_t available_memory)
{
  size_t x=available_memory;
  size_t temp=N;
  temp*=2;
  x/=temp;
  return x;
}

void preprocessing(float * data_t, int N,int L)
{
  for (int i = 0; i < N; i++)
  {
    float * row = data_t + i * L;
    double sum1 = 0, sum2 = 0;
    for (int l = 0; l < L; l++)
    {
      sum1 += row[l];
    }
    sum1 /= L;
    for (int l = 0; l < L; l++)
    {
      sum2 += (row[l] - sum1) * (row[l] - sum1);
    }
    sum2 = sqrt(sum2);
    for (int l = 0; l < L; l++)
    {
      if(sum2!=0)
        row[l] = (row[l] - sum1) / sum2;
      else
        if(sum2==0)
          row[l]=0;
    }
  }
}


int main(int argc, char *argv[])
{
  if (argc != 3) {
     std::cout<<"Usage: " << argv[0] << " <Number of voxels> <Length of time series>\n";
     return 1;
  }

  int N = atoi(argv[1]);
  int L = atoi(argv[2]);
  std::cout<<"Number of voxels: "<<N<<"  "<<"Length of time series: "<<L<<"\n\n";

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> uniform_distr (-6.f, 6.f);

  int k = 0, l = 0;
  float * data = new float [L * N];
  for (  k = 0; k < N; k++)
    for ( l = 0; l < L; l++)
      data[k*L+l] = uniform_distr(g);

  size_t M11 = (N-1);
  M11 *= N;
  M11 /= 2;

  float * upper_tri = new float [M11];

  for(size_t indii=0;indii<M11;indii++)
    upper_tri[indii]=0;

  std::cout<<"\nComputing correlations ...\n";

  const size_t free_memory = memory_budget();

  size_t app_memory = sizeof(float) * ((size_t)N * L + (size_t)N * N + M11);

  iris::Platform platform;
  device=init_iris(platform,argc,argv);
  auto start = std::chrono::steady_clock::now();

  if (app_memory < free_memory) {
    CorMat_singlePass(upper_tri, data, N, L);
  }
  else {
    CorMat_multiPass(upper_tri, data, N, L);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout<<"\nRunning time for computing correlations: \n"<< (time * 1e-9f) << " (s)\n";

  double checksum = 0;
  for(size_t tab =0;tab<M11;tab++) {
    checksum += upper_tri[tab];
  }
  std::cout<<"Checksum: " << checksum << "\n";

  if (N < 100 && L < 100) {
    std::cout<<"\nWriting correlation values into the text file ... \n";
    std::ofstream correlations_print;
    correlations_print.open("corrs.txt");
    for(size_t tab =0;tab<M11;tab++) {
      correlations_print << upper_tri[tab] << '\n';
    }
    correlations_print.close();
    std::cout<<"\nCorrelations are stored into the text file corrs.txt \n";
  }

  delete [] upper_tri;
  delete [] data;

  const int errors=platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}

int correlations(float* upper, float* data, int N, int L, bool multiple) {
  preprocessing(data,N,L);
  iris::DMem input(data,size_t(N)*L*sizeof(float));
  int remaining=N,base=0;
  size_t written=0;
  long long gemm_ns=0,extract_ns=0;
  size_t available=memory_budget()/sizeof(float)-size_t(N)*L;
  while(remaining>0) {
    int rows=multiple ? std::min<size_t>(remaining,remaining_B(remaining,available)) : remaining;
    if(rows<=0) throw std::runtime_error("Memory budget cannot hold one correlation row");
    size_t count=size_t(rows)*remaining- size_t(rows)*(rows+1)/2;
    size_t total=size_t(rows)*remaining;
    std::vector<float> matrix(total);
    iris::DMem cormat(matrix.data(),total*sizeof(float));
    // Last chunk of one voxel has no pairs; original input still normalizes it.
    if(count==0) break;
    iris::DMem output(upper+written,count*sizeof(float));
    auto start=std::chrono::steady_clock::now();
    void* p[]={ &input,&cormat,&rows,&remaining,&L,&base };
    int info[]={ iris_r,iris_w,sizeof(rows),sizeof(remaining),sizeof(L),sizeof(base) };
    launch_iris(device,"gram",((total+255)/256)*256,256,6,p,info);
    auto middle=std::chrono::steady_clock::now();
    gemm_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(middle-start).count();
    if(multiple) {
      void* p2[]={ &cormat,&output,&rows,&remaining };
      int i2[]={ iris_r,iris_w,sizeof(rows),sizeof(remaining) };
      launch_iris(device,"ker2",((total+THREAD_BLOCK_SIZE-1)/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE,THREAD_BLOCK_SIZE,4,p2,i2,&output);
    } else {
      void* p2[]={ &cormat,&output,&N };
      int i2[]={ iris_r,iris_w,sizeof(N) };
      launch_iris(device,"ker",((total+THREAD_BLOCK_SIZE-1)/THREAD_BLOCK_SIZE)*THREAD_BLOCK_SIZE,THREAD_BLOCK_SIZE,3,p2,i2,&output);
    }
    auto end=std::chrono::steady_clock::now();
    extract_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(end-middle).count();
    written+=count; base+=rows; remaining-=rows;
  }
  std::cout << (multiple ? "Total Kernel time (s)\n" : "Kernel time (s)\n")
            << "GEMM: " << gemm_ns*1e-9 << ", Extract upper triangle: " << extract_ns*1e-9 << "\n";
  return 1;
}
int CorMat_singlePass(float* upper,float* data,int N,int L) {
  return correlations(upper,data,N,L,false);
}
int CorMat_multiPass(float* upper,float* data,int N,int L) {
  return correlations(upper,data,N,L,true);
}
