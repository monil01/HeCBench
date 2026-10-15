// Actual IRIS DMem transfer and physical stream-lifetime benchmark.
// Original AMD license retained in LICENSE.
#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <memory>
#include <set>
#include <string>
#include <algorithm>
#include <cstdint>

static size_t values=0x1000;
static int serial=0;
static void ok(int rc) {
  if(rc!=IRIS_SUCCESS || iris_error_count()) {fprintf(stderr,"IRIS failure %d\n",rc);exit(2);}
}
static void hip_ok(hipError_t rc) {
  if(rc!=hipSuccess){fprintf(stderr,"HIP metadata/sync failure %s\n",hipGetErrorString(rc));exit(2);}
}
static size_t cap(const char* name,size_t original) {
  const char* v=getenv(name);if(!v)return original;
  size_t n=strtoul(v,nullptr,10);if(!n)exit(2);return std::min(n,original);
}
static void statistics(uint64_t* v) {ok(iris_hip_lifecycle_stats(0,v,0));}
static void verify_buffers(std::vector<std::unique_ptr<iris::DMem>>& buffers,
                           const std::vector<float>& input,const char* mode) {
  FILE* f=nullptr;
  if(const char* prefix=getenv("HECBENCH_SCCD_DUMP")) {
    std::string path=std::string(prefix)+"."+mode+"."+std::to_string(serial)+".bin";
    f=fopen(path.c_str(),"wb");if(!f)exit(2);
  }
  ++serial;
  std::vector<float> output(values);
  iris::DMem result(output.data(),values*sizeof(float));
  int n=static_cast<int>(values);size_t local[]={256},global[]={((values+255)/256)*256};
  uint64_t before[8],after[8];statistics(before);
  for(size_t b=0;b<buffers.size();++b) {
    iris::Task task;ok(iris_task_set_stream_index(task.task(),0));
    void* args[]={buffers[b].get(),&result,&n};int info[]={iris_r,iris_w,sizeof(n)};
    ok(task.kernel("read_copied_buffer",1,nullptr,global,local,3,args,info));
    ok(task.flush_out(result));ok(task.submit(0,nullptr,true));
    for(size_t i=0;i<values;++i)
      if(output[i]!=input[i]) {fprintf(stderr,"FAIL %s buffer%zu element%zu\n",mode,b,i);exit(1);}
    if(f && fwrite(output.data(),sizeof(float),values,f)!=values)exit(2);
  }
  statistics(after);
  // The verifier must not repair a missed copy by automatically uploading input.
  if(after[6]!=before[6] || after[7]!=before[7]) {
    fprintf(stderr,"FAIL verifier caused additional H2D copies\n");exit(1);
  }
  if(f && fclose(f))exit(2);
  printf("PASS IRIS %s: all%zu buffers x%zu floats read on GPU; verifier H2D=0\n",mode,buffers.size(),values);
}
static void run_case(int test,bool stream_mode) {
  const size_t stream_counts[]={1,2,4,8},buffer_counts[]={1,100,1000,5000};
  size_t streams=stream_counts[test%4],buffers_count=cap("HECBENCH_SCCD_BUFFER_CAP",buffer_counts[test/4]);
  size_t iterations=cap("HECBENCH_SCCD_ITER_CAP",256/(streams*(size_t(1)<<(test/4+1))));
  const char* mode=stream_mode?"Stream":"Baseline";size_t bytes=values*sizeof(float);
  std::vector<float> input(values);for(size_t i=0;i<values;++i)input[i]=1.618f+i;
  std::vector<std::vector<float>> host(buffers_count,input);
  std::vector<std::unique_ptr<iris::DMem>> buffers;
  std::set<void*> physical;
  for(size_t b=0;b<buffers_count;++b) {
    buffers.emplace_back(new iris::DMem(host[b].data(),bytes));
    void* pointer=nullptr;ok(iris_mem_arch(buffers.back()->mem(),0,&pointer));
    if(!pointer || !physical.insert(pointer).second)exit(1);
  }
  // Clear initial runtime slots outside the timed loop, never replace between
  // measured create -> copies -> sync -> destroy.
  if(stream_mode)for(size_t s=0;s<streams;++s)
    if(iris_dev_get_stream(0,1+s))ok(iris_hip_stream_destroy(0,1+s));
  uint64_t before[8],after[8];statistics(before);
  auto begin=std::chrono::steady_clock::now();
  for(size_t iteration=0;iteration<iterations;++iteration) {
    if(stream_mode)for(size_t s=0;s<streams;++s) {
      ok(iris_hip_stream_create(0,1+s,0));
      auto handle=reinterpret_cast<hipStream_t>(iris_dev_get_stream(0,1+s));unsigned flags=99;
      if(!handle)exit(1);hip_ok(hipStreamGetFlags(handle,&flags));if(flags!=0)exit(1);
    }
    for(size_t s=0;s<streams;++s) {
      int index=stream_mode?int(s+1):0;
      // Bound command batches also support the full5000-buffer default.
      for(size_t first=0;first<buffers_count;first+=128) {
        iris::Task copy;ok(iris_task_set_stream_index(copy.task(),index));
        for(size_t b=first;b<std::min(buffers_count,first+128);++b) {
          ok(iris_data_mem_refresh(buffers[b]->mem()));
          ok(copy.h2d(buffers[b].get()));
        }
        // Waiting before the next refresh prevents a queued older copy from
        // clearing freshness and turning the next requested H2D into a no-op.
        ok(copy.submit(0,nullptr,true));
      }
    }
    if(stream_mode)for(size_t s=0;s<streams;++s) {
      hip_ok(hipStreamSynchronize(reinterpret_cast<hipStream_t>(iris_dev_get_stream(0,s+1))));
      ok(iris_hip_stream_destroy(0,s+1));
      if(iris_dev_get_stream(0,s+1)!=nullptr)exit(1);
    } else {
      int device=0;ok(iris_device_synchronize(1,&device));
    }
  }
  auto end=std::chrono::steady_clock::now();statistics(after);
  uint64_t copies=iterations*streams*buffers_count,lifetimes=stream_mode?iterations*streams:0;
  if(after[0]-before[0]!=lifetimes || after[1]-before[1]!=lifetimes ||
     after[6]-before[6]!=copies || after[7]-before[7]!=copies*bytes ||
     after[2]!=before[2] || after[3]!=before[3] ||
     after[4]!=before[4] || after[5]!=before[5]) {
    fprintf(stderr,"FAIL measured lifetime/physical copy counts\n");exit(1);
  }
  printf("[%s] %zu streams %zu buffers %zu iterations %.9f ms per stream; physical create/destroy=%lu/%lu H2D=%lu bytes=%lu\n",
    mode,streams,buffers_count,iterations,std::chrono::duration<double,std::milli>(end-begin).count()/(iterations*streams),
    after[0]-before[0],after[1]-before[1],after[6]-before[6],after[7]-before[7]);
  verify_buffers(buffers,input,mode);
}
int main(int argc,char**argv) {
  if(const char* v=getenv("HECBENCH_SCCD_VALUES")){values=strtoul(v,nullptr,10);if(!values || values>size_t(INT32_MAX))return 2;}
  iris::Platform p;ok(p.init(&argc,&argv,1));
  int backend=0;ok(iris_device_info(0,iris_backend,&backend,nullptr));
  if(backend!=iris_hip)return 2;
  ok(iris_set_malloc_async(0)); // allocations are legacy, outside timed lifetime loops
  ok(iris_hip_stream_destroy(0,0));ok(iris_hip_stream_use_default(0,0));
  if(iris_dev_get_stream(0,0)!=nullptr)return 2; // genuine legacy default handle
  printf("Baseline slot0 uses HIP legacy default stream (null handle)\n");
  run_case(0,false);
  for(int test=0;test<16;++test)run_case(test,false);
  ok(iris_hip_stream_destroy(0,0)); // unbind borrowed default, never destroy it
  ok(iris_hip_stream_create(0,0,0)); // untimed validation stream
  run_case(0,true);
  for(int test=0;test<16;++test)run_case(test,true);
  return p.finalize();
}
