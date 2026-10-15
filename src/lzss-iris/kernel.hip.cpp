#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include "utils.h"

#define BLOCK_SIZE 2048    // in unit of byte, the size of one data block
#define THREAD_SIZE 128     // in unit of datatype, the size of the thread block, so as the size of symbols per iteration
#define WINDOW_SIZE 32     // in unit of datatype, maximum 255, the size of the sliding window, so as the maximum match length
#define INPUT_TYPE uint32_t // define input type
// #define DEBUG

// Define the compress match kernel functions
template <typename T>
__device__ void compressKernelI(const T *input, uint32_t numOfBlocks,
                                uint32_t *__restrict__ flagArrSizeGlobal,
                                uint32_t *__restrict__ compressedDataSizeGlobal,
                                uint8_t *__restrict__ tmpFlagArrGlobal,
                                uint8_t *__restrict__ tmpCompressedDataGlobal,
                                int minEncodeLength)
{
  // Block size in uint of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of datatype
  const uint32_t threadSize = THREAD_SIZE;

  // Allocate shared memory for the lookahead buffer of the whole block, the
  // sliding window is included
  __shared__ T buffer[blockSize];
  __shared__ uint8_t lengthBuffer[blockSize];
  __shared__ uint8_t offsetBuffer[blockSize];
  __shared__ uint32_t prefixBuffer[blockSize + 1];

  // initialize the tid
  int tid = 0;

  // Copy the memeory from global to shared
  for (int i = 0; i < blockSize / threadSize; i++)
  {
    buffer[threadIdx.x + threadSize * i] =
        input[blockIdx.x * blockSize + threadIdx.x + threadSize * i];
  }

  // Synchronize all threads to ensure that the buffer is fully loaded
  __syncthreads();

  // find match for every data point
  for (int iteration = 0; iteration < (int)(blockSize / threadSize);
       iteration++)
  {
    // Initialize the lookahead buffer and the sliding window pointers
    tid = threadIdx.x + iteration * threadSize;
    int bufferStart = tid;
    int bufferPointer = bufferStart;
    int windowStart =
        bufferStart - int(WINDOW_SIZE) < 0 ? 0 : bufferStart - WINDOW_SIZE;
    int windowPointer = windowStart;

    uint8_t maxLen = 0;
    uint8_t maxOffset = 0;
    uint8_t len = 0;
    uint8_t offset = 0;

    while (windowPointer < bufferStart && bufferPointer < blockSize)
    {
      if (buffer[bufferPointer] == buffer[windowPointer])
      {
        if (offset == 0)
        {
          offset = bufferPointer - windowPointer;
        }
        len++;
        bufferPointer++;
      }
      else
      {
        if (len > maxLen)
        {
          maxLen = len;
          maxOffset = offset;
        }
        len = 0;
        offset = 0;
        bufferPointer = bufferStart;
      }
      windowPointer++;
    }
    if (len > maxLen)
    {
      maxLen = len;
      maxOffset = offset;
    }

    lengthBuffer[threadIdx.x + iteration * threadSize] = maxLen;
    offsetBuffer[threadIdx.x + iteration * threadSize] = maxOffset;

    // initialize array as 0
    prefixBuffer[threadIdx.x + iteration * threadSize] = 0;
  }
  __syncthreads();

  // find encode information
  uint32_t flagCount = 0;
  __shared__ uint8_t byteFlagArr[(blockSize / 8)];

  if (threadIdx.x == 0)
  {
    uint8_t flagPosition = 0x01;
    uint8_t byteFlag = 0;

    int encodeIndex = 0;

    while (encodeIndex < blockSize)
    {
      // if length < minEncodeLength, no match is found
      if (lengthBuffer[encodeIndex] < minEncodeLength)
      {
        prefixBuffer[encodeIndex] = sizeof(T);
        encodeIndex++;
      }
      // if length > minEncodeLength, match is found
      else
      {
        prefixBuffer[encodeIndex] = 2;
        encodeIndex += lengthBuffer[encodeIndex];
        byteFlag |= flagPosition;
      }
      // store the flag if there are 8 bits already
      if (flagPosition == 0x80)
      {
        byteFlagArr[flagCount] = byteFlag;
        flagCount++;
        flagPosition = 0x01;
        byteFlag = 0;
        continue;
      }
      flagPosition <<= 1;
    }
    if (flagPosition != 0x01)
    {
      byteFlagArr[flagCount] = byteFlag;
      flagCount++;
    }
  }
  __syncthreads();

  // prefix summation, up-sweep
  int prefixSumOffset = 1;
  for (uint32_t d = blockSize >> 1; d > 0; d = d >> 1)
  {
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++)
    {
      tid = threadIdx.x + iteration * threadSize;
      if (tid < d)
      {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;
        prefixBuffer[bi] += prefixBuffer[ai];
      }
      __syncthreads();
    }
    prefixSumOffset *= 2;
  }

  // clear the last element
  if (threadIdx.x == 0)
  {
    // printf("block size: %d flag array size: %d\n", prefixBuffer[blockSize - 1], flagCount);
    compressedDataSizeGlobal[blockIdx.x] = prefixBuffer[blockSize - 1];
    flagArrSizeGlobal[blockIdx.x] = flagCount;
    prefixBuffer[blockSize] = prefixBuffer[blockSize - 1];
    prefixBuffer[blockSize - 1] = 0;
  }
  __syncthreads();

  // prefix summation, down-sweep
  for (int d = 1; d < blockSize; d *= 2)
  {
    prefixSumOffset >>= 1;
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++)
    {
      tid = threadIdx.x + iteration * threadSize;

      if (tid < d)
      {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;

        uint32_t t = prefixBuffer[ai];
        prefixBuffer[ai] = prefixBuffer[bi];
        prefixBuffer[bi] += t;
      }
      __syncthreads();
    }
  }

  // encoding phase one
  int tmpCompressedDataGlobalOffset;
  tmpCompressedDataGlobalOffset = blockSize * blockIdx.x * sizeof(T);
  for (int iteration = 0; iteration < (int)(blockSize / threadSize); iteration++)
  {
    tid = threadIdx.x + iteration * threadSize;
    if (prefixBuffer[tid + 1] != prefixBuffer[tid])
    {
      if (lengthBuffer[tid] < minEncodeLength)
      {
        uint32_t tmpOffset = prefixBuffer[tid];
        uint8_t *bytePtr = (uint8_t *)&buffer[tid];
        for (int tmpIndex = 0; tmpIndex < sizeof(T); tmpIndex++)
        {
          tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + tmpIndex] = *(bytePtr + tmpIndex);
        }
      }
      else
      {
        uint32_t tmpOffset = prefixBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset] = lengthBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + 1] = offsetBuffer[tid];
      }
    }
  }

  // Copy the memeory back
  if (threadIdx.x == 0)
  {
    for (int flagArrIndex = 0; flagArrIndex < flagCount; flagArrIndex++)
    {
      tmpFlagArrGlobal[blockSize / 8 * blockIdx.x + flagArrIndex] = byteFlagArr[flagArrIndex];
    }
  }
}
// Define the compress Encode kernel functions
template <typename T>
__device__ void compressKernelIII(uint32_t numOfBlocks,
                                  const uint32_t *__restrict__ flagArrOffsetGlobal,
                                  const uint32_t *__restrict__ compressedDataOffsetGlobal,
                                  const uint8_t *__restrict__ tmpFlagArrGlobal,
                                  const uint8_t *__restrict__ tmpCompressedDataGlobal,
                                  uint8_t *__restrict__ flagArrGlobal,
                                  uint8_t *__restrict__ compressedDataGlobal)
{
  // Block size in uint of bytes
  const int blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of bytes
  const int threadSize = THREAD_SIZE;

  // find block index
  int blockIndex = blockIdx.x;

  int flagArrOffset = flagArrOffsetGlobal[blockIndex];
  int flagArrSize = flagArrOffsetGlobal[blockIndex + 1] - flagArrOffsetGlobal[blockIndex];

  int compressedDataOffset = compressedDataOffsetGlobal[blockIndex];
  int compressedDataSize = compressedDataOffsetGlobal[blockIndex + 1] - compressedDataOffsetGlobal[blockIndex];

  int tid = threadIdx.x;

  while (tid < flagArrSize)
  {
    flagArrGlobal[flagArrOffset + tid] = tmpFlagArrGlobal[blockSize / 8 * blockIndex + tid];
    tid += threadSize;
  }

  tid = threadIdx.x;

  while (tid < compressedDataSize)
  {
    compressedDataGlobal[compressedDataOffset + tid] = tmpCompressedDataGlobal[blockSize * sizeof(T) * blockIndex + tid];
    tid += threadSize;
  }
}

// Define the decompress kernel functions
template <typename T>
__device__ void decompressKernel(T *output, uint32_t numOfBlocks,
                                 const uint32_t *__restrict__ flagArrOffsetGlobal,
                                 const uint32_t *__restrict__ compressedDataOffsetGlobal,
                                 const uint8_t *__restrict__ flagArrGlobal,
                                 const uint8_t *__restrict__ compressedDataGlobal)
{
  // Block size in unit of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < numOfBlocks)
  {
    int flagArrOffset = flagArrOffsetGlobal[tid];
    int flagArrSize = flagArrOffsetGlobal[tid + 1] - flagArrOffsetGlobal[tid];

    int compressedDataOffset = compressedDataOffsetGlobal[tid];

    uint32_t dataPointsIndex = 0;
    uint32_t compressedDataIndex = 0;

    uint8_t byteFlag;

    for (int flagArrayIndex = 0; flagArrayIndex < flagArrSize; flagArrayIndex++)
    {
      byteFlag = flagArrGlobal[flagArrOffset + flagArrayIndex];

      for (int bitCount = 0; bitCount < 8; bitCount++)
      {
        int matchFlag = (byteFlag >> bitCount) & 0x1;
        if (matchFlag == 1)
        {
          int length = compressedDataGlobal[compressedDataOffset + compressedDataIndex];
          int offset = compressedDataGlobal[compressedDataOffset + compressedDataIndex + 1];
          compressedDataIndex += 2;
          int dataPointsStart = dataPointsIndex;
          for (int tmpDecompIndex = 0; tmpDecompIndex < length; tmpDecompIndex++)
          {
            output[tid * blockSize + dataPointsIndex] = output[tid * blockSize + dataPointsStart - offset + tmpDecompIndex];
            dataPointsIndex++;
          }
        }
        else
        {
          uint8_t *tmpPtr = (uint8_t *)&output[tid * blockSize + dataPointsIndex];
          for (int tmpDecompIndex = 0; tmpDecompIndex < sizeof(T); tmpDecompIndex++)
          {
            *(tmpPtr + tmpDecompIndex) = compressedDataGlobal[compressedDataOffset + compressedDataIndex + tmpDecompIndex];
          }

          compressedDataIndex += sizeof(T);
          dataPointsIndex++;
        }
        if (dataPointsIndex >= blockSize)
        {
          return;
        }
      }
    }
  }
}
extern "C" __global__ void compress_match(const uint32_t* in, uint32_t blocks,
 uint32_t* flagsizes, uint32_t* sizes, uint8_t* tmpflags, uint8_t* tmpdata, int minimum) {
 compressKernelI<uint32_t>(in,blocks,flagsizes,sizes,tmpflags,tmpdata,minimum);
}
extern "C" __global__ void compress_pack(uint32_t blocks, const uint32_t* flags,
 const uint32_t* sizes, const uint8_t* tmpflags, const uint8_t* tmpdata, uint8_t* outflags, uint8_t* outdata) {
 compressKernelIII<uint32_t>(blocks,flags,sizes,tmpflags,tmpdata,outflags,outdata);
}
extern "C" __global__ void decompress(uint32_t* out, uint32_t blocks,
 const uint32_t* flags, const uint32_t* offsets, const uint8_t* flagdata, const uint8_t* data) {
 decompressKernel<uint32_t>(out,blocks,flags,offsets,flagdata,data);
}
extern "C" __global__ void exclusive_scan(const uint32_t* input, uint32_t* output, uint32_t n) {
 if (threadIdx.x || blockIdx.x) return;
 uint32_t total=0;
 for (uint32_t i=0;i<n;++i) { output[i]=total;total+=input[i]; }
}
