// NVIDIA sample semantics, adapted to IRIS DMem/tasks. Original source license
// is retained in LICENSE; no native allocation/copy/launch owns device memory.
#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>

static void checked(int rc) {
  if (rc != IRIS_SUCCESS || iris_error_count()) {
    fprintf(stderr, "IRIS failure %d\n", rc); exit(2);
  }
}
static void hip_checked(hipError_t rc) {
  if (rc != hipSuccess) { fprintf(stderr, "HIP control failure %s\n", hipGetErrorString(rc)); exit(2); }
}
static int bounded(const char* name, int def) {
  const char* v=getenv(name); int n=v?atoi(v):def;
  if(n<1) { fprintf(stderr,"invalid %s\n",name);exit(2); } return n;
}
static void dump(const std::vector<float>& c, const char* name, bool timing) {
  if(const char* prefix=getenv("HECBENCH_SOA_DUMP")) {
    std::string path=std::string(prefix)+"."+name+(timing?".timing":".check");
    FILE* f=fopen(path.c_str(),"wb");
    if(!f || fwrite(c.data(),sizeof(float),c.size(),f)!=c.size() || fclose(f)) exit(2);
  }
}
static void stats(uint64_t* v, bool reset=false) { checked(iris_hip_lifecycle_stats(0,v,reset)); }

static void run_policy(int mode, bool timing, int count, const std::vector<float>& a,
                       const std::vector<float>& b, std::vector<float>& c) {
  const char* names[]={"basicAllocation","basicStreamOrderedAllocation","streamOrderedAllocationPostSync"};
  const char* name=names[mode]; int index=mode?1:0, n=int(a.size());size_t bytes=a.size()*sizeof(float);
  printf("Starting %s()\n",name);
  checked(iris_set_malloc_async(mode?1:0));
  if(mode) {
    checked(iris_hip_stream_destroy(0,index));
    checked(iris_hip_stream_create(0,index,1));
  }
  uint64_t threshold=0;
  hipMemPool_t pool;
  hip_checked(hipDeviceGetDefaultMemPool(&pool,0));
  if(timing && mode) {
    threshold=mode==2?uint64_t(ULONG_MAX):0;
    hip_checked(hipMemPoolSetAttribute(pool,hipMemPoolAttrReleaseThreshold,&threshold));
  }
  hip_checked(hipMemPoolGetAttribute(pool,hipMemPoolAttrReleaseThreshold,&threshold));
  if(mode && timing && threshold!=(mode==2?uint64_t(ULONG_MAX):0)) exit(2);
  uint64_t before[8],after[8];stats(before);
  auto begin=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i) {
    auto A=new iris::DMem(const_cast<float*>(a.data()),bytes);
    auto B=new iris::DMem(const_cast<float*>(b.data()),bytes);
    auto C=new iris::DMem(c.data(),bytes);
    {
      iris::Task task;
      checked(iris_task_set_stream_index(task.task(),index));
      checked(task.h2d(A));checked(task.h2d(B));
      void* args[]={A,B,C,&n};int info[]={iris_r,iris_r,iris_w,sizeof(n)};
      size_t local[]={256},global[]={((a.size()+255)/256)*256};
      checked(task.kernel("vectorAddGPU",1,nullptr,global,local,4,args,info));
      checked(task.submit(0,nullptr,true));
    }
    delete A;delete B; // actual hipFree/hipFreeAsync through runtime ownership
    {
      iris::Task read;
      checked(iris_task_set_stream_index(read.task(),index));
      checked(read.flush_out(*C));checked(read.submit(0,nullptr,true));
    }
    delete C;
    // Complete queued frees as in the original per-iteration synchronization.
    hip_checked(hipStreamSynchronize(reinterpret_cast<hipStream_t>(iris_dev_get_stream(0,index))));
    checked(IRIS_SUCCESS);
  }
  auto end=std::chrono::steady_clock::now();
  stats(after);
  uint64_t delta[8];for(int k=0;k<8;++k)delta[k]=after[k]-before[k];
  uint64_t allocations=uint64_t(3)*count;
  if(delta[mode?2:4]!=allocations || delta[mode?3:5]!=allocations ||
     delta[mode?4:2] || delta[mode?5:3] || delta[6]!=uint64_t(2)*count ||
     delta[7]!=uint64_t(2)*count*bytes) {
    fprintf(stderr,"FAIL physical policy counts %s: async=%lu/%lu legacy=%lu/%lu h2d=%lu/%lu\n",
      name,delta[2],delta[3],delta[4],delta[5],delta[6],delta[7]);exit(1);
  }
  // Stronger diagnostic check in every phase, including original timed phases.
  float errorNorm=0,refNorm=0;
  for(size_t j=0;j<c.size();++j) {
    float ref=a[j]+b[j],diff=c[j]-ref;
    if(!std::isfinite(c[j]) || c[j]!=ref) {fprintf(stderr,"FAIL element%zu\n",j);exit(1);}
    errorNorm+=diff*diff;refNorm+=ref*ref;
  }
  if(!(sqrt(errorNorm)/sqrt(refNorm)<1e-6))exit(1);
  printf("%s PASSED all%zu values; physical async=%lu/%lu legacy=%lu/%lu h2d=%lu bytes=%lu threshold=%lu\n",
    name,c.size(),delta[2],delta[3],delta[4],delta[5],delta[6],delta[7],threshold);
  if(timing)printf("Total elapsed time: %f (us) over %d iterations\n",
    std::chrono::duration<double,std::micro>(end-begin).count(),count);
  dump(c,name,timing);
  if(mode) {
    checked(iris_hip_stream_destroy(0,index));
    if(iris_dev_get_stream(0,index)!=nullptr)exit(2);
    checked(iris_hip_stream_create(0,index,1)); // restore slot for later phases
  }
}
int main(int argc,char** argv) {
  int n=bounded("HECBENCH_SOA_ELEMENTS",33554432),count=bounded("HECBENCH_SOA_REPEATS",100);
  std::vector<float>a(n),b(n),c(n);
  // Native rand() default seed/sequence, before runtime initialization.
  for(int i=0;i<n;++i){a[i]=rand()/(float)RAND_MAX;b[i]=rand()/(float)RAND_MAX;}
  iris::Platform platform;checked(platform.init(&argc,&argv,1));
  int backend=0;checked(iris_device_info(0,iris_backend,&backend,nullptr));
  if(backend!=iris_hip){fprintf(stderr,"Requires HIP-only IRIS\n");return 2;}
  for(int timing=0;timing<2;++timing)for(int mode=0;mode<3;++mode)
    run_policy(mode,timing,count,a,b,c);
  checked(iris_set_malloc_async(0));
  return platform.finalize();
}
