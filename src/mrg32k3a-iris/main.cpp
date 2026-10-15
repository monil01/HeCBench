// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// Original algorithm: ../mrg32k3a-cuda/main.cu. Redistribution terms: LICENSE.
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "validation.hpp"
static void check(int status) {
  if(status) { std::fprintf(stderr,"MRG operation failed: %d\n",status); std::abort(); }
}
static constexpr unsigned long long seed=1234ULL,offset=0ULL;
static constexpr hiprandOrdering_t order=HIPRAND_ORDERING_PSEUDO_BEST;
struct GenerationStats { size_t calls=0,values=0; };
static int generate(void* raw_stream,unsigned,void*const* args) {
  auto* data=hecbench_vendor_arg<float*>(args,1);
  size_t n=hecbench_vendor_arg<size_t>(args,2);
  auto* stats=hecbench_vendor_arg<GenerationStats*>(args,3);
  hipPointerAttribute_t attributes{};
  int device=-1;
  if(hipGetDevice(&device)!=hipSuccess ||
     hipPointerGetAttributes(&attributes,data)!=hipSuccess ||
     attributes.type!=hipMemoryTypeDevice || attributes.isManaged ||
     attributes.device!=device) return 5;
  hiprandGenerator_t gen=nullptr;
  if(hiprandCreateGenerator(&gen,HIPRAND_RNG_PSEUDO_MRG32K3A)!=HIPRAND_STATUS_SUCCESS) return 1;
  auto stream=static_cast<hipStream_t>(raw_stream);
  int status=0;
  if(hiprandSetGeneratorOffset(gen,offset)!=HIPRAND_STATUS_SUCCESS ||
     hiprandSetGeneratorOrdering(gen,order)!=HIPRAND_STATUS_SUCCESS ||
     hiprandSetPseudoRandomGeneratorSeed(gen,seed)!=HIPRAND_STATUS_SUCCESS ||
     hiprandSetStream(gen,stream)!=HIPRAND_STATUS_SUCCESS ||
     hiprandGenerateUniform(gen,data,n)!=HIPRAND_STATUS_SUCCESS) status=2;
  if(hipStreamSynchronize(stream)!=hipSuccess) status=3;
  if(hiprandDestroyGenerator(gen)!=HIPRAND_STATUS_SUCCESS) status=4;
  if(!status) { ++stats->calls; stats->values+=n; }
  return status;
}
static void run_on_device(std::vector<float>& result,GenerationStats& stats) {
  // As in the source, every call allocates a fresh output and a fresh generator.
  iris::DMem mem(result.data(),result.size()*sizeof(float));
  size_t n=result.size(); hecbench_vendor_callback cb=generate;
  GenerationStats* counters=&stats;
  void* args[]={&cb,&mem,&n,&counters}; int info[]={sizeof(cb),iris_w,sizeof(n),sizeof(counters)};
  size_t gws[]={1},lws[]={1}; iris::Task task;
  check(task.kernel("hecbench_vendor",1,nullptr,gws,lws,4,args,info));
  check(task.flush_out(mem)); check(task.submit(0,nullptr,1));
  mrg_trace(result,"device");
}
static void run_on_host(std::vector<float>& result) {
  // This is the original timed CPU hipRAND workload/reference, NOT a fallback.
  hiprandGenerator_t gen=nullptr;
  check(hiprandCreateGeneratorHost(&gen,HIPRAND_RNG_PSEUDO_MRG32K3A));
  check(hiprandSetGeneratorOffset(gen,offset));
  check(hiprandSetGeneratorOrdering(gen,order));
  check(hiprandSetPseudoRandomGeneratorSeed(gen,seed));
  check(hiprandGenerateUniform(gen,result.data(),result.size()));
  check(hiprandDestroyGenerator(gen));
  mrg_trace(result,"host");
}
int main(int argc,char**argv) {
  if(argc!=3) {
    std::printf("Usage: %s <number of pseudorandom numbers to generate> <repeat>\n",argv[0]);
    return 1;
  }
  int n=std::atoi(argv[1]),repeat=std::atoi(argv[2]);
  if(n<1 || repeat<1) return 1;
  std::vector<float> host(n,0),device(n,-1);
  mrg_trace(host,"host",true); mrg_trace(device,"device",true);
  setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);
  iris::Platform platform; check(platform.init(&argc,&argv,1));
  GenerationStats stats;
  run_on_host(host); run_on_device(device,stats);
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;++i) run_on_host(host);
  auto end=std::chrono::steady_clock::now();
  std::printf("Average execution time on host: %f (us)\n",
      std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-3/repeat);
#ifdef DEBUG
  std::puts("Host"); for(size_t i=host.size()>10?host.size()-10:0;i<host.size();++i) std::printf("%0.6f\n",host[i]); std::puts("=====");
#endif
  start=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;++i) run_on_device(device,stats);
  end=std::chrono::steady_clock::now();
  std::printf("Average execution time on device: %f (us)\n",
      std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-3/repeat);
#ifdef DEBUG
  std::puts("Device"); for(size_t i=device.size()>10?device.size()-10:0;i<device.size();++i) std::printf("%0.6f\n",device[i]); std::puts("=====");
#endif
  check(stats.calls!=size_t(repeat)+1 || stats.values!=(size_t(repeat)+1)*size_t(n));
  check(iris_error_count()); check(platform.finalize());
  bool pass=true;
  for(int i=0;i<n;++i)
    if(!std::isfinite(device[i]) || device[i]<=0 || device[i]>1 ||
       std::fabs(host[i]-device[i])>1e-3f) pass=false;
  std::printf("IRIS hipRAND MRG32K3A generations: %zu; values: %zu; seed: %llu; offset: %llu\n",stats.calls,stats.values,seed,offset);
  std::puts(pass?"PASS":"FAIL"); return pass?0:1;
}
