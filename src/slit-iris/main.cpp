/***********************************************************
 * Developed for Seminar in Parallelisation of Physics
 * Calculations on GPUs with CUDA, Department of Physics
 * Technical University of Munich. Author: Binu Amaratunga
 * IRIS DMem/hipFFT adaptation.
 ***********************************************************/
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <rocfft/rocfft.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../slit-cuda/reference.h"
static void check(int status) {
  if(status) { std::fprintf(stderr,"slit operation failed: %d\n",status); std::abort(); }
}
struct FFTState { hipfftHandle plan; double nanoseconds=0; int calls=0; int n; size_t work_bytes=0; };
static int transform(void* raw_stream,unsigned,void*const* args) {
  auto* state=hecbench_vendor_arg<FFTState*>(args,1);
  int operation=hecbench_vendor_arg<int>(args,2);
  if(operation==0) {
    if(hipfftCreate(&state->plan)!=HIPFFT_SUCCESS ||
       hipfftSetAutoAllocation(state->plan,0)!=HIPFFT_SUCCESS) return 4;
    return int(hipfftMakePlan2d(state->plan,state->n,state->n,HIPFFT_Z2Z,&state->work_bytes));
  }
  if(operation==2) {
    if(hipfftDestroy(state->plan)!=HIPFFT_SUCCESS) return 5;
    return int(rocfft_cleanup());
  }
  auto* data=hecbench_vendor_arg<hipfftDoubleComplex*>(args,3);
  auto* work=hecbench_vendor_arg<void*>(args,4);
  auto stream=static_cast<hipStream_t>(raw_stream);
  if(hipfftSetStream(state->plan,stream)!=HIPFFT_SUCCESS ||
     hipfftSetWorkArea(state->plan,work)!=HIPFFT_SUCCESS) return 1;
  auto start=std::chrono::steady_clock::now();
  auto status=hipfftExecZ2Z(state->plan,data,data,HIPFFT_FORWARD);
  if(status!=HIPFFT_SUCCESS) {
    std::fprintf(stderr,"hipfftExecZ2Z failed: %d\n",int(status)); return 2;
  }
  if(hipStreamSynchronize(stream)!=hipSuccess) return 3;
  state->nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now()-start).count();
  ++state->calls;
  return 0;
}
static void control(FFTState& state,int operation) {
  FFTState* context=&state; hecbench_vendor_callback cb=transform;
  void* args[]={&cb,&context,&operation};
  int info[]={sizeof(cb),sizeof(context),sizeof(operation)};
  size_t gws[]={1},lws[]={1}; iris::Task task;
  check(task.kernel("hecbench_vendor",1,nullptr,gws,lws,3,args,info));
  check(task.submit(0,nullptr,1));
}
int main(int argc,char**argv) {
  if(argc!=3) {
    std::printf("Usage: %s <the transform size in the x and y dimensions> <repeat>\n",argv[0]);
    return 1;
  }
  const int N=std::atoi(argv[1]),repeat=std::atoi(argv[2]);
  if(N<1 || (N&(N-1)) || repeat<1) return 1;
  const size_t count=size_t(N)*N,bytes=count*sizeof(hipfftDoubleComplex);
  std::printf("Running FFT for %d x %d = %zu = 2 ^ %d data points...\n",
      N,N,count,int(std::log(double(count))/std::log(2.0)));
  std::vector<hipfftDoubleComplex> input(count),data(count);
  std::vector<double> input_ref(count),output(count),output_ref(count);
  for(int j=0;j<N;++j) for(int i=0;i<N;++i) {
    const bool slit=std::abs(i-N/2)<=10 && std::abs(i-N/2)>=8 && std::abs(j-N/2)<=4;
    input[size_t(j)*N+i]={slit?1.0:0.0,0.0};
    input_ref[size_t(j)*N+i]=slit?1.0:0.0;
  }
  setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);
  iris::Platform platform; check(platform.init(&argc,&argv,1));
  FFTState state{}; state.n=N; control(state,0);
  const size_t work_bytes=state.work_bytes;
  {
    std::vector<unsigned char> workspace(work_bytes?work_bytes:1);
    iris::DMem mem(data.data(),bytes),work(workspace.data(),workspace.size());
    for(int iteration=0;iteration<repeat;++iteration) {
      std::memcpy(data.data(),input.data(),bytes);
      check(iris_data_mem_refresh(mem.mem()));
      iris::Task upload; check(upload.h2d(&mem));
      check(upload.submit(0,nullptr,1));
      FFTState* context=&state; hecbench_vendor_callback cb=transform;
      int operation=1;
      void* args[]={&cb,&context,&operation,&mem,&work};
      int info[]={sizeof(cb),sizeof(context),sizeof(operation),iris_rw,iris_w};
      size_t gws[]={1},lws[]={1}; iris::Task fft;
      check(fft.kernel("hecbench_vendor",1,nullptr,gws,lws,5,args,info));
      check(fft.submit(0,nullptr,1));
    }
    iris::Task download; check(download.flush_out(mem));
    check(download.submit(0,nullptr,1)); control(state,2);
  }
  check(iris_error_count()); check(platform.finalize()); check(state.calls!=repeat);
  std::printf("Average execution time of FFT: %lf ms\n",state.nanoseconds*1e-6/repeat);
  std::printf("IRIS hipFFT executions: %d; DMem workspace bytes: %zu\n",state.calls,work_bytes);
  for(size_t i=0;i<count;++i) output[i]=data[i].x*data[i].x+data[i].y*data[i].y;
  reference(input_ref.data(),output_ref.data(),N);
  bool pass=true;
  for(size_t i=0;i<count;++i)
    if(!std::isfinite(output[i]) || std::fabs(output[i]-output_ref[i])>1e-3) pass=false;
  if(const char* path=std::getenv("HECBENCH_SLIT_DUMP")) {
    FILE* f=std::fopen(path,"wb"); if(!f) return 2;
    const bool written=std::fwrite(data.data(),sizeof(data[0]),count,f)==count &&
        std::fwrite(output.data(),sizeof(double),count,f)==count;
    if(std::fclose(f) || !written) return 2;
  }
  std::puts(pass?"PASS":"FAIL"); return pass?0:1;
}
