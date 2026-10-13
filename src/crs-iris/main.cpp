//  Created by Liu Chengjian on 17/10/9.
//  Copyright (c) 2017 csliu. All rights reserved.
//

#include <iris/iris.hpp>
inline void iris_check(int rc) { if(rc != IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc); exit(1); } }
#include "GCRSMatrix.h"
#include "utils.h"
#include "dispatch.hpp"

int main(int argc, char * argv[]) {

  if (argc != 3) {
    printf("Usage: ./%s workSizePerDataParityBlockInMB numberOfTasks\n", argv[0]);
    exit(0);
  }

  int workSize = atoi(argv[1]);
  size_t bufSize = (size_t)workSize * 1024 * 1024; // workSize per data parity block

  int taskNum = atoi(argv[2]);
  if (taskNum <= 0) {
    printf("Error: Number of tasks must be a positive number\n");
    return 1;
  }

  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,1));
  double encode_time = 0.0;

#ifdef DUMP
  for (int m = 4; m <= 4; ++m) {
  for (int n = 8; n <= 8; ++n) {  // w is updated in the nested loop
  for (int k = MAX_K; k <= MAX_K; ++k) {
#else
  for (int m = 1; m <= 4; ++m) {
  for (int n = 4; n <= 8; ++n) {  // w is updated in the nested loop
  for (int k = m; k <= MAX_K; ++k) {
#endif

    int w = gcrs_check_k_m_w(k, m, n);
    if (w < 0) continue;

#ifdef DUMP
    printf("k:%d, m:%d w:%d\n",k,m,w);
#endif

    int *bitmatrix = gcrs_create_bitmatrix(k, m, w);
    //printMatrix(bitmatrix, k*w, m*w);

    //  adjust the bufSize
    size_t bufSizePerTask = align_value(bufSize / taskNum, sizeof(long) * w);
    bufSize = bufSizePerTask * taskNum;

    //  compute the bufSize for the last task
    size_t bufSizeForLastTask = bufSize - (bufSizePerTask * (taskNum - 1));
#ifdef DUMP
    printf("Total Size:%zu Size per task:%zu Size for last task:%zu\n",
           bufSize, bufSizePerTask, bufSizeForLastTask);
#endif

    // allocate host buffers
    char* data = (char*) malloc (bufSize * k);
    char* code = (char*) malloc (bufSize * m);

    // initialize host buffer
    generateRandomValue(data, bufSize * k);

    // taskSize will determine the number of kernels to run on a device
    int taskSize = 1;
    int mRemain = m;

    // adjust taskSize
    if (m >= MAX_M) {
      taskSize = m / MAX_M;
      if (m % MAX_M != 0) ++taskSize;
    }

#ifdef DUMP
    printf("task size: %d\n", taskSize);
#endif

    // set up kernel execution parameters
    int *mValue = (int*) malloc (sizeof(int) * taskSize);
    int *index = (int*) malloc (sizeof(int) * taskSize);
    coding_func *coding_function_ptrs = (coding_func*) malloc (sizeof(coding_func) * taskSize);

    for (int i = 0; i < taskSize; ++i) {
      if (mRemain < MAX_M) {
        mValue[i] = mRemain;
      }else{
        mValue[i] = MAX_M;
        mRemain = mRemain - MAX_M;
      }

      if (i == 0) {
        index[i] = 0;
      }else{
        index[i] = index[i-1] + k * w;
      }
      coding_function_ptrs[i] = coding_func_array[(mValue[i] - 1) * (MAX_W - MIN_W + 1)+ w - MIN_W];
    }

    //  create and then update encoding bit matrix
    unsigned int *all_columns_bitmatrix = (unsigned int*) malloc (sizeof(unsigned int) * k * w * taskSize);

    int mValueSum = 0;
    for (int i = 0; i < taskSize; ++i) {

      unsigned int *column_bitmatrix = gcrs_create_column_coding_bitmatrix(
          k, mValue[i], w, bitmatrix + k * w * mValueSum * w);

      memcpy((all_columns_bitmatrix + i * k * w), column_bitmatrix, k * w * sizeof(unsigned int));

      free(column_bitmatrix);
      mValueSum += mValue[i];
    }

    iris::DMem d_bitmatrix(all_columns_bitmatrix,sizeof(unsigned int)*k*w*taskSize);

    int warpThreadNum = 32;
    int threadNum = MAX_THREAD_NUM;
    size_t workSizePerWarp = warpThreadNum / w * w;
    size_t workSizePerBlock = threadNum / warpThreadNum * workSizePerWarp * sizeof(size_t);
    size_t blockNum = bufSizePerTask / workSizePerBlock;

    if ((bufSizePerTask % workSizePerBlock) != 0) {
      blockNum = blockNum + 1;
    }

#ifdef DUMP
    printf("#blocks: %zu  blockSize: %d\n", blockNum, threadNum);
#endif

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < taskNum; ++i) {
      int count = (i == taskNum-1) ? bufSizeForLastTask : bufSizePerTask;

      iris::DMem data_slice(data+i*k*bufSizePerTask,k*count);

      int workSizePerGrid = count / sizeof(long);
      int size = workSizePerGrid * sizeof(long);
      mValueSum = 0;
      for (int j = 0; j < taskSize; ++j) {
        iris::DMem code_slice(code+i*m*bufSizePerTask+mValueSum*size,mValue[j]*size);
        iris::Task task;
        void *args[] = { &k, &index[j], &data_slice, &code_slice, &d_bitmatrix, &workSizePerGrid };
        int info[] = { sizeof(k), sizeof(index[j]), iris_r, iris_rw, iris_r, sizeof(workSizePerGrid) };
        size_t gws[] = { blockNum*size_t(threadNum) }, lws[] = { size_t(threadNum) };
        iris_check(task.kernel(coding_function_ptrs[j],1,nullptr,gws,lws,6,args,info));
        iris_check(task.flush_out(code_slice));
        iris_check(task.submit(iris_gpu,nullptr,true));
        mValueSum += mValue[j];
      }

    }
    auto end = std::chrono::steady_clock::now();
    auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#ifdef DUMP
    printf("Encoding time over %d tasks: %lf (ms)\n", taskNum, etime * 1e-6);
#endif
    encode_time += etime * 1e-6;

  if (getenv("CRS_VERIFY")) {
    unsigned long long input_hash=1469598103934665603ULL,code_hash=input_hash;
    for(size_t b=0;b<bufSize*size_t(k);++b) input_hash=(input_hash ^ (unsigned char)data[b])*1099511628211ULL;
    for(size_t b=0;b<bufSize*size_t(m);++b) code_hash=(code_hash ^ (unsigned char)code[b])*1099511628211ULL;
    printf("VERIFY k=%d m=%d w=%d bytes=%zu input=%016llx code=%016llx\n",k,m,w,bufSize,input_hash,code_hash);
  }


#ifdef DUMP
    for (int i = 0; i < bufSize*m; i++) printf("%d\n", code[i]);
    printf("\n");
#endif



    free(mValue);
    free(index);
    free(coding_function_ptrs);
    free(bitmatrix);
    free(all_columns_bitmatrix);


    free(code);
    free(data);
  }
  }
  }

  printf("Total encoding time %lf (s)\n", encode_time * 1e-3);

  iris_check(platform.finalize());
  return 0;
}
