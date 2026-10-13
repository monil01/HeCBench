#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include <iris/iris.hpp>
#include <limits>
#include "reference.h"
#include "code.h"

int main(int argc, char* argv[])
{
  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS)return 1;
  bool gpu=false;for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }if(!gpu)return 1;
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(uint8_t);
  const size_t code_size = 256 * sizeof(float); // code.h

  std::vector<float> A(n);
  std::vector<uint8_t> out(n), ref(n);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{0.f, 1.f};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen); 
    ref[i] = dQuantize<0>(code, 0.f, A[i]);
  }

  if(n>size_t(std::numeric_limits<int>::max()) || n==0 || repeat<1)return 1;
  int kernel_n=int(n);
  iris::DMem d_A(A.data(),A_size),d_code(code,code_size),d_out(out.data(),out_size);

  const int block_size = 256;

  dim3 grid ((n+block_size-1)/block_size);
  dim3 block (block_size);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    iris::Task task;size_t gws[]={size_t(grid.x)*block.x},lws[]={block.x};
    void* args[]={&d_code,&d_A,&d_out,&kernel_n};int info[]={iris_r,iris_r,iris_w,sizeof(kernel_n)};
    if(task.kernel("kQuantize",1,nullptr,gws,lws,4,args,info)!=IRIS_SUCCESS)return 2;
    if(i==repeat-1)task.flush_out(d_out);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kQuantize kernel with block size %d: %f (us)\n",
          block_size, (time * 1e-3f) / repeat);
  
  if(const char* path=getenv("HECBENCH_ARTIFACT")){FILE* f=fopen(path,"wb");if(!f)return 3;fwrite(out.data(),1,out_size,f);fclose(f);}

  printf("%s\n", out == ref ? "PASS" : "FAIL");



  int errors=platform.error_count();platform.finalize();return out==ref && !errors ? 0 : 1;
}
