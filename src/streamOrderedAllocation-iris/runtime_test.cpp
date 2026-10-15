// Regression test for incremental IRIS runtime lifecycle patch04.
#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void ok(int rc) { if(rc!=IRIS_SUCCESS || iris_error_count()){fprintf(stderr,"IRIS regression failure %d\n",rc);exit(1);} }
static void reject(int rc) { if(rc!=IRIS_ERROR){fprintf(stderr,"Expected control rejection\n");exit(1);} }
int main(int argc,char**argv) {
  iris::Platform p;ok(p.init(&argc,&argv,1));
  reject(iris_set_malloc_async(-1));reject(iris_set_malloc_async(2));
  reject(iris_hip_stream_destroy(-1,2));reject(iris_hip_stream_create(0,-1,0));
  reject(iris_hip_stream_destroy(0,99999));reject(iris_hip_lifecycle_stats(0,nullptr,0));
  ok(iris_hip_stream_destroy(0,2));
  for(int kind=0;kind<3;++kind) {
    uint64_t before[8],after[8];ok(iris_hip_lifecycle_stats(0,before,1));
    ok(iris_set_malloc_async(1));ok(iris_hip_stream_create(0,2,0));
    reject(iris_hip_stream_create(0,2,0));
    auto handle=reinterpret_cast<hipStream_t>(iris_dev_get_stream(0,2));
    if(!handle || hipStreamQuery(handle)!=hipSuccess)return 1;
    int n=259;size_t bytes=n*sizeof(float);std::vector<float>a(n),b(n),c(n);
    for(int i=0;i<n;++i){a[i]=i*.5f;b[i]=1.25f;}
    auto A=new iris::DMem(a.data(),bytes),B=new iris::DMem(b.data(),bytes),C=new iris::DMem(c.data(),bytes);
    ok(iris_dmem_set_hip_allocation_kind(C->mem(),kind));
    iris::Task t;ok(iris_task_set_stream_index(t.task(),2));
    ok(t.h2d(A));ok(t.h2d(B));void*args[]={A,B,C,&n};int info[]={iris_r,iris_r,iris_w,sizeof(n)};
    size_t local[]={256},global[]={512};ok(t.kernel("vectorAddGPU",1,nullptr,global,local,4,args,info));
    ok(t.flush_out(*C));ok(t.submit(0,nullptr,true));
    for(int i=0;i<n;++i)if(c[i]!=a[i]+b[i])return 1;
    void*pointer=nullptr;ok(iris_mem_arch(C->mem(),0,&pointer));hipPointerAttribute_t attrs;
    if(hipPointerGetAttributes(&attrs,pointer)!=hipSuccess ||
       (kind==0 && (attrs.type!=hipMemoryTypeDevice || attrs.isManaged)) ||
       (kind==1 && !attrs.isManaged) || (kind==2 && attrs.type!=hipMemoryTypeHost))return 1;
    // Stream destruction must reject live async allocations (A and B always).
    reject(iris_hip_stream_destroy(0,2));
    // Free must follow recorded allocation mode/stream, not this global flag.
    ok(iris_set_malloc_async(0));delete A;delete B;delete C;
    if(hipStreamSynchronize(handle)!=hipSuccess)return 1;
    ok(iris_hip_stream_destroy(0,2));if(iris_dev_get_stream(0,2))return 1;
    reject(iris_hip_stream_destroy(0,2));
    ok(iris_hip_lifecycle_stats(0,after,0));
    uint64_t expected=kind==0?3:2;
    if(after[0]!=1 || after[1]!=1 || after[2]!=expected || after[3]!=expected ||
       after[4] || after[5] || after[6]!=2 || after[7]!=2*bytes)return 1;
    printf("PASS kind%d all259 outputs and physical pointer kind; create/destroy=1/1, async=%lu/%lu after policy-off, H2D=2; invalid/live-slot controls rejected\n",kind,after[2],after[3]);
  }
  return p.finalize();
}
