#include <chrono>
#include <iostream>
#include <cstring>
#include <iris/iris.hpp>
#include "gpu_lock.hpp"
#include "config.hpp"

#include "reference.h"

#ifdef DEBUG
void dump (const char* work_path, const char* result_filename, const float* result) {
  char file_name[100];
  int i;

  FILE *fp;

  sprintf(file_name, "%s/%s", work_path, result_filename);
  if (!(fp = fopen(file_name, "w"))) {
    printf("File %s cannot be opened for write.\n", result_filename);
    exit(-1);
  }
  for (i = 0; i < SAMPLE_TEST_LEN; ++i)
    fprintf(fp, "%f\n", result[i]);
  fclose(fp);
}
#endif

void init(const char* work_path, const char* input_filename, const char* weight_filename,
    float* sample_input, float* inW, float* intW, float* intB, float* outW, float* outB) {

  char file_name[100];

  float weightVal;

  int i, j, k;

  FILE *fp;

  sprintf(file_name, "%s/%s", work_path, input_filename);
  // Read in sample input from "input.hpp" file
  if (!(fp = fopen(file_name, "r"))) {
    printf("File %s cannot be opened for read.\n", input_filename);
    exit(-1);
  }

  for (i = 0; i < SAMPLE_TEST_LEN; ++i) {
    fscanf(fp, "%f", &sample_input[i]);
  }
  fclose(fp);

  // duplicate the sample using the first sample
  for (int i = 1; i < N; i++)
    memcpy(sample_input+i*SAMPLE_TEST_LEN, sample_input, SAMPLE_TEST_LEN*sizeof(float));

  // Load weights and perform inference for LSTM 1.
  sprintf(file_name, "%s/%s", work_path, weight_filename);
  if (!(fp = fopen(file_name, "r"))) {
    printf("File %s cannot be opened for read.\n", weight_filename);
    exit(-1);
  }
  for (j = 0; j < 4; ++j) {
    for (i = 0; i < 5; ++i) {
      fscanf(fp, "%f", &weightVal);
      inW[j*5+i] = weightVal;
    }
  }
  for (k = 0; k < 4; ++k) {
    for (j = 0; j < 5; ++j) {
      for (i = 0; i < 5; ++i) {
        fscanf(fp, "%f", &weightVal);
        intW[k*25+j*5+i] = weightVal;
      }
    }
  }
  for (j = 0; j < 4; ++j) {
    for (i = 0; i < 5; ++i) {
      fscanf(fp, "%f", &weightVal);
      intB[j*5+i] = weightVal;
    }
  }
  for (i = 0; i < 5; ++i) {
    fscanf(fp, "%f", &weightVal);
    outW[i] = weightVal;
  }
  fscanf(fp, "%f", &weightVal);
  *outB = weightVal;
  fclose(fp);
}

long lstm_n5(const float* x,
             const float* inW,
             const float* intW,
             const float* intB,
             const float* outW,
             const float* outB,
                   float* y)
{
  GpuLock lock;
  iris::DMem mx(const_cast<float*>(x),size_t(N)*SAMPLE_TEST_LEN*4),
   mi(const_cast<float*>(inW),80),mw(const_cast<float*>(intW),400),
   mb(const_cast<float*>(intB),80),mo(const_cast<float*>(outW),20),
   mob(const_cast<float*>(outB),4),my(y,size_t(N)*SAMPLE_TEST_LEN*4);
  iris::Task task;
  void *params[]={&mx,&mi,&mw,&mb,&mo,&mob,&my};
  int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_w};
  size_t global=N,local=WGS;
  auto start=std::chrono::steady_clock::now();
  task.kernel("lstm_inference",1,nullptr,&global,&local,7,params,info);
  task.flush_out(my);
  if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("IRIS task failed");
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  iris::Platform platform;{GpuLock lock;platform.init(&argc,&argv,1);}
  int devices=0;iris_device_count(&devices);
  for(int d=0;d<devices;d++){int b=0;size_t z=0;iris_device_info(d,iris_backend,&b,&z);printf("IRIS backend=%d HIP=%d batches=%d sequence=%d\n",b,iris_hip,N,SAMPLE_TEST_LEN);}
  bool all_ok=true;
  const int repeat = atoi(argv[1]);

  float* sample_input = (float*) aligned_alloc(64, sizeof(float)*N*SAMPLE_TEST_LEN);
  float* infer1_out = (float*) aligned_alloc(64, sizeof(float)*N*SAMPLE_TEST_LEN);
  float* infer2_out = (float*) aligned_alloc(64, sizeof(float)*N*SAMPLE_TEST_LEN);

  float inW[20], intW[100], intB[20], outW[5];
  float outB;

  const char* work_path = "./";
  const char* input_filename = "input.hpp";
  const char* weight1_filename = "weight_1.hpp";
  const char* weight2_filename = "weight_2.hpp";
#ifdef DEBUG
  const char* result1_filename = "float_infer_result_1.hpp";
  const char* result2_filename = "float_infer_result_2.hpp";
#endif

  long kernel_time = 0;
  for (int n = 0; n < repeat; n++) {
    init(work_path, input_filename, weight1_filename, sample_input, inW, intW, intB, outW, &outB) ;
    auto start = std::chrono::steady_clock::now();
    kernel_time += lstm_n5(sample_input, inW, intW, intB, outW, &outB, infer1_out);
    auto end = std::chrono::steady_clock::now();
    auto elapsedTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Device offload time: " <<  elapsedTime << " ms\n";

    if (n == 0)
      all_ok &= reference(sample_input, inW, intW, intB, outW, &outB, infer1_out);

#ifdef DEBUG
    dump(work_path, result1_filename, infer1_out);
#endif

    init(work_path, input_filename, weight2_filename, sample_input, inW, intW, intB, outW, &outB) ;
    start = std::chrono::steady_clock::now();
    kernel_time += lstm_n5(sample_input, inW, intW, intB, outW, &outB, infer2_out);
    end = std::chrono::steady_clock::now();
    elapsedTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Device offload time: " <<  elapsedTime << " ms\n";

    if (n == 0)
      all_ok &= reference(sample_input, inW, intW, intB, outW, &outB, infer2_out);

#ifdef DEBUG
    dump(work_path, result2_filename, infer2_out);
#endif
  }
  std::cout << "Average kernel time: " <<  kernel_time * 1e-6 / (2 * repeat) << " ms\n";

  free(sample_input);
  free(infer1_out);
  free(infer2_out);
  printf("Processing complete.\n");
  {GpuLock lock;platform.finalize();}
  return all_ok?0:1;
}
