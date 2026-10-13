#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>
#include "iris_support.h"
#include <cstdint>
#include <cmath>
static int device;
static int64_t spatial_dim(const char* name,int64_t fallback) {
  const char* value=std::getenv(name);
  int64_t result=value?std::strtoll(value,nullptr,10):fallback;
  if(result<1) throw std::runtime_error("Spatial dimensions must be positive");
  return result;
}
static void dump(const char* name,const std::vector<float>& data) {
  if(!std::getenv("HECBENCH_DUMP")) return;
  FILE* fp=fopen(name,"wb");
  if(!fp || fwrite(data.data(),sizeof(float),data.size(),fp)!=data.size())
    throw std::runtime_error("Artifact write failed");
  fclose(fp);
}


using namespace std::chrono;

void Forward(int repeat)
{
  int64_t ndims = 5;
  int64_t size = 5;
  float alpha = 0.000122;
  float beta = 0.750000;
  float k = 1.000000;
  int64_t N = 6;
  int64_t C = 150;
  int64_t D = spatial_dim("HECBENCH_LRN_D",100);
  int64_t H = spatial_dim("HECBENCH_LRN_H",160);
  int64_t W = spatial_dim("HECBENCH_LRN_W",160);
  int64_t stride_mb = C*D*H*W;
  int64_t wk_size = N*C*D*H*W;

  std::vector<float> src(wk_size, 0);
  std::vector<float> dst(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++) { 
    src[i] = rand() / (float)RAND_MAX;
  }

  size_t bytes_to_copy_d=wk_size*sizeof(float);
  iris::DMem src_mem(src.data(),bytes_to_copy_d), dst_mem(dst.data(),bytes_to_copy_d);
  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size = wg_size * 2) {

    int64_t wg_cnt = (wk_size + wg_size - 1) / wg_size;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      size_t lws=wg_size, gws=wg_cnt*lws;
      iris::Task task;
      void* args[]={&src_mem,&dst_mem,&N,&C,&D,&H,&W,&stride_mb,&ndims,&wk_size,&size,&alpha,&beta,&k};
      int info[]={iris_r,iris_rw,sizeof(N),sizeof(C),sizeof(D),sizeof(H),sizeof(W),sizeof(stride_mb),sizeof(ndims),sizeof(wk_size),sizeof(size),sizeof(alpha),sizeof(beta),sizeof(k)};
      iris_check(task.kernel("lrn_fwd_kernel",1,nullptr,&gws,&lws,14,args,info),"forward");
      iris_check(task.submit(device,nullptr,true),"HIP forward");
    }

    auto stop = high_resolution_clock::now();

    auto time = (duration_cast<microseconds>(stop - start)).count()/1e6f;
    printf("Average execution time of lrn_fwd_kernel: %.6f sec \n", time / repeat);

    auto data_inGB = (2 * wk_size * sizeof(float)) / 1e9f;
    auto bandwidth = data_inGB * repeat / time;

    printf("Kernel bandwidth: %.6f GB/s \n", bandwidth);
  }

  iris::Task flush;
  iris_check(flush.flush_out(dst_mem),"flush destination");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");
  dump("lrn_forward.bin",dst);

  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) { 
    checksum += dst[i];
  }
  printf("Checksum: %lf\n", checksum / wk_size);

}


void Backward(int repeat)
{
  int64_t ndims = 5;
  int64_t size = 5;
  float alpha = 0.000122;
  float beta = 0.750000;
  float k = 1.000000;
  int64_t N = 5;
  int64_t C = 150;
  int64_t D = spatial_dim("HECBENCH_LRN_D",100);
  int64_t H = spatial_dim("HECBENCH_LRN_H",160);
  int64_t W = spatial_dim("HECBENCH_LRN_W",160);
  int64_t stride_mb = C*D*H*W;
  int64_t wk_size = N*C*D*H*W;

  std::vector<float> src(wk_size, 0);
  std::vector<float> dst(wk_size, 0);
  std::vector<float> diff_src(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++) { 
    diff_src[i] = src[i] = rand() / (float)RAND_MAX;
  }

  size_t bytes_to_copy_d=wk_size*sizeof(float);
  dst=src; // Native backward dst device buffer is initialized from src.
  iris::DMem src_mem(src.data(),bytes_to_copy_d), dst_mem(dst.data(),bytes_to_copy_d);
  iris::DMem diff_src_mem(diff_src.data(),bytes_to_copy_d);
  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size = wg_size * 2) {

    int64_t wg_cnt = (wk_size + wg_size - 1) / wg_size;

    auto start = high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      size_t lws=wg_size, gws=wg_cnt*lws;
      iris::Task task;
      void* args[]={&src_mem,&dst_mem,&diff_src_mem,&N,&C,&D,&H,&W,&stride_mb,&ndims,&wk_size,&size,&alpha,&beta,&k};
      int info[]={iris_r,iris_r,iris_rw,sizeof(N),sizeof(C),sizeof(D),sizeof(H),sizeof(W),sizeof(stride_mb),sizeof(ndims),sizeof(wk_size),sizeof(size),sizeof(alpha),sizeof(beta),sizeof(k)};
      iris_check(task.kernel("lrn_bwd_kernel",1,nullptr,&gws,&lws,15,args,info),"backward");
      iris_check(task.submit(device,nullptr,true),"HIP backward");
    }

    auto stop = high_resolution_clock::now();

    auto time = (duration_cast<microseconds>(stop - start)).count()/1e6f;
    printf("Average execution time of lrn_bwd_kernel: %.6f sec \n", time / repeat);

    auto data_inGB = (3 * wk_size * sizeof(float)) / 1e9f;
    auto bandwidth = data_inGB * repeat / time;

    printf("Kernel bandwidth: %.6f GB/s \n", bandwidth);
  }


  iris::Task flush;
  iris_check(flush.flush_out(dst_mem),"flush destination");
  iris_check(flush.flush_out(diff_src_mem),"flush gradient");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");
  dump("lrn_backward_dst.bin",dst);
  dump("lrn_gradient.bin",diff_src);

  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) { 
    checksum += dst[i];
  }
  printf("Checksum: %lf\n", checksum / wk_size);
}


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  if(repeat<1) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  device=iris_hip_device();

  Forward(repeat);
  Backward(repeat);

  iris_check(platform.finalize(),"finalize");
  return 0;
}
