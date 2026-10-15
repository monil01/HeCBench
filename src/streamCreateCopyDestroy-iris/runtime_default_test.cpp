// Regression for borrowed legacy-default slots, distinct from owned streams.
#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void ok(int rc){if(rc!=IRIS_SUCCESS || iris_error_count())exit(1);}
static void reject(int rc){if(rc!=IRIS_ERROR)exit(1);}
int main(int argc,char**argv){
 iris::Platform p;ok(p.init(&argc,&argv,1));ok(iris_set_malloc_async(0));
 reject(iris_hip_stream_use_default(-1,0));reject(iris_hip_stream_use_default(0,-1));
 reject(iris_hip_stream_use_default(0,99999));reject(iris_hip_stream_use_default(0,0));
 uint64_t before[8],after[8];ok(iris_hip_lifecycle_stats(0,before,1));
 ok(iris_hip_stream_destroy(0,0));ok(iris_hip_stream_use_default(0,0));
 if(iris_dev_get_stream(0,0)!=nullptr || hipStreamQuery(nullptr)!=hipSuccess)return 1;
 reject(iris_hip_stream_use_default(0,0));reject(iris_hip_stream_create(0,0,0));
 int n=263;std::vector<float>input(n),output(n);for(int i=0;i<n;++i)input[i]=i*.5f+1.25f;
 {
  iris::DMem source(input.data(),n*sizeof(float)),result(output.data(),n*sizeof(float));
  iris::Task t;ok(iris_task_set_stream_index(t.task(),0));ok(iris_data_mem_refresh(source.mem()));
  ok(t.h2d(&source));void*args[]={&source,&result,&n};int info[]={iris_r,iris_w,sizeof(n)};
  size_t local[]={256},global[]={512};ok(t.kernel("read_copied_buffer",1,nullptr,global,local,3,args,info));
  ok(t.flush_out(result));ok(t.submit(0,nullptr,true));
  for(int i=0;i<n;++i)if(input[i]!=output[i])return 1;
 }
 ok(iris_hip_stream_destroy(0,0)); // borrowed binding only, not hipStreamDestroy(null)
 ok(iris_hip_lifecycle_stats(0,after,0));
 if(after[0] || after[1]!=1 || after[6]!=1 || after[7]!=n*sizeof(float))return 1;
 reject(iris_hip_stream_destroy(0,0));ok(iris_hip_stream_create(0,0,0));
 if(!iris_dev_get_stream(0,0))return 1;
 printf("PASS borrowed legacy-default null slot: all263 GPU-read values, physical H2D1/%zu bytes, no create/default-destroy; invalid/occupied/rebind controls rejected; owned stream restored\n",n*sizeof(float));
 return p.finalize();
}
