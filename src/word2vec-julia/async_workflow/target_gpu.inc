#include "cbow.h"
#include <stdio.h>
#include <assert.h>

__constant__ real expTable[EXP_TABLE_SIZE];

extern real *syn0;
extern int * table;
extern int vocab_size, layer1_size , layer1_size_aligned;
extern int negative , window;
extern int table_size;
// To batch data to minimize data transfer, sen stores words + alpha values
// alpha value start at offset = MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH

extern int * sen;

real * d_syn0 = NULL;
real* target_exp=nullptr;
real * d_syn1neg = NULL;
int  * d_sen = NULL;
unsigned int * d_random = NULL;
int * d_table = NULL;

int maxThreadsPerBlock = 256;
int numBlock;
int shared_mem_usage;

#define cudaCheck(err) { \
  if (err != cudaSuccess) { \
    printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    assert(err == cudaSuccess); \
  } \
}
void cleanUpGPU(){
  observe_final();
  cudaCheck(cudaFree(target_exp));
  cudaCheck(cudaFree(d_syn1neg));
  cudaCheck(cudaFree(d_syn0));
  cudaCheck(cudaFreeHost(sen));
  cudaCheck(cudaFree(d_sen));
  cudaCheck(cudaFree(d_random));
  cudaCheck(cudaFree(d_table));
}
void initializeGPU(){

  real * h_expTable = (real *)malloc((EXP_TABLE_SIZE ) * sizeof(real));
  for (int i = 0; i < EXP_TABLE_SIZE; i++) {
    h_expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    h_expTable[i] = h_expTable[i] / (h_expTable[i] + 1);
  }

  cudaCheck(cudaMemcpyToSymbol(expTable, h_expTable, sizeof(real) * EXP_TABLE_SIZE));
  cudaCheck(cudaMalloc((void**)&target_exp,4000));
  cudaCheck(cudaMemcpy(target_exp,h_expTable,4000,cudaMemcpyHostToDevice));
  free(h_expTable);

  if (negative>0) {
    int syn1neg_size = vocab_size * layer1_size_aligned;
    cudaCheck(cudaMalloc((void**) & d_syn1neg, syn1neg_size * sizeof(real)));
    broker(0,0,0,syn1neg_size / maxThreadsPerBlock + 1);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
  }

  int syn0_size = vocab_size * layer1_size_aligned;
  cudaCheck(cudaMalloc((void**) &d_syn0, syn0_size * sizeof(real)));
  cudaCheck(cudaMemcpy(d_syn0, syn0, syn0_size * sizeof(real), cudaMemcpyHostToDevice));

  cudaCheck(cudaMallocHost((void**)&sen, (MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int) ));
  cudaCheck(cudaMalloc((void**)&d_sen, (MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int) ));

  cudaCheck(cudaMalloc((void**) &d_random, MAX_SENTENCE_LENGTH * sizeof(unsigned int)));

  int h_random[MAX_SENTENCE_LENGTH];

  srand(123);
  for (int i = 0 ; i < MAX_SENTENCE_LENGTH; i++)
    h_random[i] = (unsigned int) rand();

  cudaCheck(cudaMemcpy(d_random, h_random, MAX_SENTENCE_LENGTH * sizeof(unsigned int), cudaMemcpyHostToDevice));

  cudaCheck(cudaMalloc((void**) & d_table, table_size * sizeof(int)));
  cudaMemcpy(d_table, table, table_size * sizeof(int), cudaMemcpyHostToDevice);

  memset(sen,0,SEN_WORDS*4);
  cudaCheck(cudaMemset(d_sen,0,SEN_WORDS*4));
  numBlock = MAX_SENTENCE_LENGTH / (BLOCK_SIZE/THREADS_PER_WORD) + 1;
  shared_mem_usage = (BLOCK_SIZE + (BLOCK_SIZE/THREADS_PER_WORD) * layer1_size_aligned * 2) * sizeof(real);
  observe_initial();
}

void TransferDataToGPU(){
  cudaCheck(cudaMemcpy(d_sen, sen,
        (MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + MAX_SENTENCE_NUM) * sizeof(int) , cudaMemcpyHostToDevice));
}

void GetResultFromGPU(){
  cudaCheck(cudaMemcpy(syn0, d_syn0, vocab_size * layer1_size_aligned * sizeof(real), cudaMemcpyDeviceToHost));
}

void TrainGPU(int sentence_num) {
  int slot=trace_submitted.fetch_add(1);
  if(slot>=TRACE_CAP){fprintf(stderr,"TRACE_OVERFLOW\n");abort();}
  TransferDataToGPU();

  broker(1,sentence_num,slot,numBlock);

#if defined(DEBUG)
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
#endif
}
