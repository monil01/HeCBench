#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <iris/iris.hpp>
#include "backend.h"

#include "svd_host.h"

void runDevice(float* input, float* output, int n, int repeat)
{
  iris::DMem answer(output,21*sizeof(float)*n), data(input,9*sizeof(float)*n);
  int threads=256, pblks=int(n/threads)+1;
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;i++) {
    iris::Task task; void *p[]={&data,&answer,&n}; int info[]={iris_r,iris_w,sizeof(n)};
    size_t g[]={size_t(pblks)*threads},l[]={size_t(threads)};
    task.kernel("svd3_SOA",1,nullptr,g,l,3,p,info); task.flush_out(answer);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
  }
  auto end=std::chrono::steady_clock::now();
  printf("Average kernel execution time: %f (us)\n",std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-3/repeat);

}

void svd3x3_ref(float* input, float* output, int testsize)
{
  for (int tid = 0; tid < testsize; tid++) 
    svd(
      input[tid + 0 * testsize], input[tid + 1 * testsize], input[tid + 2 * testsize],
      input[tid + 3 * testsize], input[tid + 4 * testsize], input[tid + 5 * testsize],
      input[tid + 6 * testsize], input[tid + 7 * testsize], input[tid + 8 * testsize],

      output[tid + 0 * testsize], output[tid + 1 * testsize], output[tid + 2 * testsize],
      output[tid + 3 * testsize], output[tid + 4 * testsize], output[tid + 5 * testsize],
      output[tid + 6 * testsize], output[tid + 7 * testsize], output[tid + 8 * testsize],
      output[tid + 9 * testsize], output[tid + 10 * testsize], output[tid + 11 * testsize],
      output[tid + 12 * testsize], output[tid + 13 * testsize], output[tid + 14 * testsize],
      output[tid + 15 * testsize], output[tid + 16 * testsize], output[tid + 17 * testsize],
      output[tid + 18 * testsize], output[tid + 19 * testsize], output[tid + 20 * testsize]
    );
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path to file> <repeat>\n";
    return 1;
  }

  iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
  // Load data
  const char* filename = argv[1];
  const int repeat = atoi(argv[2]);

  std::ifstream myfile (filename);
  if (!myfile.is_open()) {
    std::cout << "ERROR: failed to open " << filename << std::endl;
    return -1;
  }

  int testsSize;
  myfile >> testsSize;
  std::cout << "dataset size: " << testsSize << std::endl;
  if (testsSize <= 0) {
    std::cout << "ERROR: invalid dataset size\n";
    return -1;
  }

  float* input = (float*)malloc(sizeof(float) * 9 * testsSize);

  // host and device results
  float* result = (float*)malloc(sizeof(float) * 21 * testsSize);
  float* result_h = (float*)malloc(sizeof(float) * 21 * testsSize);

  int count = 0;
  for (int i = 0; i < testsSize; i++)
    for (int j = 0; j < 9; j++) myfile >> input[count++];
  myfile.close();

  // SVD 3x3 on a GPU 
  runDevice(input, result, testsSize, repeat);

  // run CPU 3x3 to verify results
  bool ok = true;
  svd3x3_ref(input, result_h, testsSize);

  for (int i = 0; i < testsSize; i++)
  {
    if (fabsf(result[i] - result_h[i]) > 1e-3) {
      std::cout << result[i] << " " << result_h[i] << std::endl;
      ok = false;
      break;
    }
  }

  if (ok)
    std::cout << "PASS\n";
  else
    std::cout << "FAIL\n";

  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *f=fopen(path,"wb"); if(!f) exit(4);
    if(fwrite(result,sizeof(float),21*testsSize,f)!=size_t(21*testsSize)) exit(4);
    if(fwrite(result_h,sizeof(float),21*testsSize,f)!=size_t(21*testsSize)) exit(4);
    fclose(f);
  }
  free(input);
  free(result);
  free(result_h);
  platform.finalize();
  return 0;
}
