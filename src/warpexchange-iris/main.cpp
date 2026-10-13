#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include "verification.h"
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}


constexpr int items_per_thread = 4;
constexpr int warp_threads = 32;

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t n = (size_t)nrows * ncols;
  const size_t A_size = n * sizeof(int);
  const size_t out_size = n * sizeof(int);

  int *A = (int*) malloc (A_size);
  int *out = (int*) malloc (out_size);

  for (size_t i = 0; i < n; i++) {
    A[i] = i;
  }

  if(nrows<=0||ncols<=0||repeat<=0||n>INT_MAX-1024)throw std::runtime_error("positive dimensions/repeat and kernel int-size limit required");
  std::memset(out,0,out_size);
  iris::Platform platform;check(platform.init(&argc,&argv,true));
  const int device=hip_device(), count=static_cast<int>(n);
  {
  iris::DMem d_A(A,A_size),d_out(out,out_size);
  size_t lws=256,gws=((n+255)/256)*256;
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    void* args[]={&d_A,&d_out,arg(count)};int info[]={iris_r,iris_w,sizeof(count)};
    iris::Task task;check(task.kernel("warp_exchange",1,nullptr,&gws,&lws,3,args,info));check(task.submit(device,nullptr,true));
  }


  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

  iris::Task flush;check(flush.flush_out(d_out));check(flush.submit(device,nullptr,true));

#ifdef DEBUG
  for (size_t i = 0; i < n; i++) {
    printf("%zu: %d %d\n", i, A[i], out[i]);
  }
#endif

  // TODO
  // verify the mapping between input and output

  // input:  0,1,2,3,4,5 ...
  // output: 0,32,64,96,1,33,...
  const int items_per_warp = items_per_thread * warp_threads;
  if (n % items_per_warp == 0) {
    size_t k;
    bool ok = true;
    for (k = 0; k < n; k += items_per_warp) {
      size_t i = k;
      for (int j = 0; j < items_per_thread; j++) {
        for (int m = 0; m < warp_threads-1; m++) {
          if (i + (m+1)*items_per_thread < n) {
            if (out[i + (m+1)*items_per_thread] - out[i + m*items_per_thread] != 1) {
              printf("Error at index %zu\n", i + (m+1)*items_per_thread);
              ok = false;
              goto stop;
            }
          }
        }
        i++;
      }
    }
    stop:
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  if(!verify_mapping(out,n))throw std::runtime_error("exact permutation verification failed");
  }
  check(platform.finalize());
  free(A);
  free(out);
  return 0;
}
