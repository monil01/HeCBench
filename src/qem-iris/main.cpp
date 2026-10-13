#include <iris/iris.hpp>
#include <cstdlib>
#include <string>
static void dump_minimum(const char* suffix, const float* data,int n) {
  if(const char* prefix=getenv("HECBENCH_QEM_OUTPUT")) {
    std::string path=std::string(prefix)+suffix;
    FILE* fp=fopen(path.c_str(),"wb");
    if(!fp || fwrite(data,sizeof(float),n,fp)!=static_cast<size_t>(n)) exit(2);
    fclose(fp);
  }
}
#include <algorithm>
#include <chrono> // for high_resolution_clock
#include <cstdio>
#include <random>

#include "reference.h"
#include "gpu_solver.h"

void generate_data(int size, int min, int max, float *data) {
  std::mt19937_64 generator{1993764};
  std::uniform_int_distribution<> dist{min, max};
  for (int i = 0; i < size; ++i) {
    data[i] = dist(generator);
  }
}


int gpu_device=-1;
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  // number of functions
  int N = getenv("HECBENCH_QEM_N") ? atoi(getenv("HECBENCH_QEM_N")) : 1999999;
  if(N<=0) return 1;
  printf("N = %d\n", N);

  float *A, *B, *C, *D, *E;
  float *minimum_ref, *minimum;

  A = static_cast<float*>(malloc(N*sizeof(float)));
  B = static_cast<float*>(malloc(N*sizeof(float)));
  C = static_cast<float*>(malloc(N*sizeof(float)));
  D = static_cast<float*>(malloc(N*sizeof(float)));
  E = static_cast<float*>(malloc(N*sizeof(float)));
  minimum_ref = static_cast<float*>(malloc(N*sizeof(float)));
  minimum = static_cast<float*>(malloc(N*sizeof(float)));

  printf("generating data...\n");

  generate_data(N, -100, 100, A);
  generate_data(N, -100, 100, B);
  generate_data(N, -100, 100, C);
  generate_data(N, -100, 100, D);
  generate_data(N, -100, 100, E);

  for (int i = 0; i < N; i++) {
    if (A[i] == 0) {
      A[i] = 1;
    } // avoid undefined behaviour in solver when A=0
  }

  float dur = 0;
  float avg = 0;
  bool ok;

  printf("####################### Reference #############\n");

  for (int k = 0; k < repeat; ++k) {
    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumCPU(N, A, B, C, D, E, minimum_ref);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  avg = 0;

  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  gpu_device=device;
  printf("####################### GPU (no streams) #############\n");

  for (int k = 0; k < repeat; ++k) {

    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumGPU(N, A, B, C, D, E, minimum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if(!ok) return 1;
  dump_minimum("-nostreams.bin",minimum,N);

  avg = 0;

  printf("####################### GPU (streams) #############\n");

  for (int k = 0; k < repeat; ++k) {

    auto start = std::chrono::high_resolution_clock::now();

    QuarticMinimumGPUStreams(N, A, B, C, D, E, minimum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    dur = elapsed.count() * 1000;
    // printf("Time (ms): %f\n", dur);
    avg += dur;
  }

  printf("Execution time (ms): %f\n", avg / repeat);

  ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(minimum[i] - minimum_ref[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if(!ok) return 1;
  dump_minimum("-streams.bin",minimum,N);

  platform.finalize();
  free(A);
  free(B);
  free(C);
  free(D);
  free(E);
  free(minimum_ref);
  free(minimum);

  return 0;
}
