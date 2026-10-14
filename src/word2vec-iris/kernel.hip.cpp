#include <hip/hip_runtime.h>
#include "cbow.h"
__device__ void reduceInWarp(volatile float * f, int idInWarp){

  for (unsigned int i=THREADS_PER_WORD /2; i>32; i>>=1) {
    if (idInWarp < i) {
      f[idInWarp] += f[idInWarp + i];
    }
    __syncthreads();
  }
  if (idInWarp < 32){
    f[idInWarp] += f[idInWarp + 32];
    f[idInWarp] += f[idInWarp + 16];
    f[idInWarp] += f[idInWarp + 8];
    f[idInWarp] += f[idInWarp + 4];
    f[idInWarp] += f[idInWarp + 2];
    f[idInWarp] += f[idInWarp + 1];
  }
}

extern "C" __global__ void device_cbow(
    const int sentence_num,
    const int layer1_size,
    const int layer1_size_aligned,
    const int window,
    const int negative,
    const int table_size,
    const int vocab_size,
    const int *__restrict__ d_sen,
    const int *__restrict__ d_table,
    float *__restrict__ d_syn0,
    float *__restrict__ d_syn1neg,
    unsigned int *__restrict__ d_random,
    const real *__restrict__ expTable, float *__restrict__ scratch, const int scratch_stride)
{
  int sentence_position = (threadIdx.x / THREADS_PER_WORD) + (blockDim.x / THREADS_PER_WORD) * blockIdx.x;
  int idInWarp = threadIdx.x % THREADS_PER_WORD;

  float *shared = scratch + blockIdx.x * scratch_stride;
  float * f = shared + (threadIdx.x / THREADS_PER_WORD) * THREADS_PER_WORD;
  float * neu1 = shared + BLOCK_SIZE + (threadIdx.x / THREADS_PER_WORD) * layer1_size_aligned;
  float * neu1e= shared + BLOCK_SIZE + (blockDim.x / THREADS_PER_WORD) * layer1_size_aligned +
                 (threadIdx.x / THREADS_PER_WORD) * layer1_size_aligned;

  if (sentence_position < MAX_SENTENCE_LENGTH) {
    unsigned int next_random = d_random[sentence_position];

    for (int sentence_idx = 0; sentence_idx < sentence_num; sentence_idx++) {

      for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD) {
        neu1[c] = 0;
        neu1e[c] = 0;
      }

      next_random = next_random * (unsigned int) 1664525 + 1013904223;
      int b = next_random % window;
      int word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + sentence_position];
      // in -> hidden
      int cw = 0;
      for (int a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {
          int w = sentence_position - window + a;
          if (w < 0 || w>= MAX_SENTENCE_LENGTH) continue;
          int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];
          for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
            neu1[c] += d_syn0[c + last_word * layer1_size_aligned];

          cw++;
        }

      if (cw) {
        for (int c = idInWarp; c < layer1_size; c+= THREADS_PER_WORD)
          neu1[c] /= cw;

        // NEGATIVE SAMPLING
        int target, label;
        float alpha =*((float *) &d_sen[MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH + sentence_idx]);

        if (negative > 0)

          for (int d = 0; d < negative + 1; d++) {

            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned int) 1664525 + 1013904223;
              target = d_table[(next_random) % table_size];
              if (target == 0)
                target = next_random % (vocab_size - 1) + 1;
              if (target == word)
                continue;
              label = 0;
            }
            int l2 = target * layer1_size_aligned;
            f[idInWarp] = 0;


            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD){
              f[idInWarp] += neu1[c] * d_syn1neg[c + l2];
            }
            __syncthreads();

            // Do reduction here;
            reduceInWarp(f, idInWarp);

            __syncthreads();

            float g;
            if (f[0] > MAX_EXP)
              g = (label - 1) * alpha;
            else if (f[0] < -MAX_EXP)
              g = (label - 0) * alpha;
            else
              g = (label - expTable[(int) ((f[0] + MAX_EXP)
                    * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
              neu1e[c] += g * d_syn1neg[c + l2];
            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
              d_syn1neg[c + l2] += g * neu1[c];
          }

        // hidden -> in
        for (int a = b; a < window * 2 + 1 - b; a++)
          if (a != window) {
            int w = sentence_position - window + a;
            if (w < 0)
              continue;
            if (w >= MAX_SENTENCE_LENGTH)
              continue;
            int last_word = d_sen[sentence_idx * MAX_SENTENCE_LENGTH + w];

            for (int c = idInWarp; c < layer1_size; c+=THREADS_PER_WORD)
              d_syn0[c + last_word * layer1_size_aligned] += neu1e[c];
          }
      }
    }// End for sentence_idx

    // Update d_random
    if (idInWarp == 0 ) d_random[sentence_position] = next_random;
  }
}
