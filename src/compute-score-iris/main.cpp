// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without ion, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <iris/iris.hpp>
#include <vector>
using uint=unsigned int;
using ulong=unsigned long;
static_assert(sizeof(ulong)==8);
#include "options.h"
#include "scoped_ptrs.h"
using namespace aocl_utils;

#define MANUAL_VECTOR      8 
#define NUM_THREADS_PER_WG 64
#define BLOOM_1            5 
#define BLOOM_2            0x7FFFF
#define BLOOM_SIZE         14
#define docEndingTag       0xFFFFFFFF

// Params
uint block_size = 64;
uint repeat = 100;
uint total_num_docs = 256*1024;
uint total_doc_size = 0;
uint total_doc_size_no_padding = 0;

// Host Buffers
scoped_aligned_ptr<uint> h_docWordFrequencies_dimm1;
scoped_aligned_ptr<uint> h_docWordFrequencies_dimm2;
scoped_aligned_ptr<ulong> h_profileWeights;
scoped_aligned_ptr<ulong> h_docInfo;
scoped_aligned_ptr<uint> h_isWordInProfileHash;
scoped_aligned_ptr<uint> h_startingDocID;
scoped_aligned_ptr<uint> h_numItemsPerThread;
scoped_aligned_ptr<ulong> h_profileScore;
scoped_aligned_ptr<uint> h_docSizes;

static uint m_z = 1;
static uint m_w = 1;
static uint rand_desh()
{
  m_z = 36969 * (m_z & 65535) + (m_z >> 16);
  m_w = 18000 * (m_w & 65535) + (m_w >> 16);
  return (m_z << 16) + m_w;
}

double sampleNormal() 
{     
  double u = ((double) rand() / (RAND_MAX)) * 2 - 1;     
  double v = ((double) rand() / (RAND_MAX)) * 2 - 1;     
  double r = u * u + v * v;     
  if (r == 0 || r > 1) return sampleNormal();     
  double c = sqrt(-2 * log(r) / r);     
  return u * c; 
} 

#define DOC_LEN_SIGMA 100
#define AVG_DOC_LEN   350

uint get_doc_length()
{
  int len = sampleNormal() * DOC_LEN_SIGMA + AVG_DOC_LEN;
  if (len < 10) { len = 10; } // Arbitray lower bound;
  return (uint) len;
}

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

void setupData()
{
  h_startingDocID.reset( total_num_docs );
  h_numItemsPerThread.reset( total_num_docs );
  h_profileScore.reset( total_num_docs );
  h_docInfo.reset( total_num_docs );

  h_docSizes.reset( total_num_docs );

  total_doc_size = 0; 	
  total_doc_size_no_padding = 0;

  for (uint i=0; i<total_num_docs; i++) {
    uint unpadded_size = get_doc_length();
    uint size = unpadded_size & (~(2*block_size-1));
    if (unpadded_size & ((2*block_size-1))) size += 2*block_size; // Multiple of block_size
    h_startingDocID[i] = total_doc_size/2;
    h_numItemsPerThread[i] = size / (2*block_size);

    ulong start_line = total_doc_size / (2*block_size);
    ulong end_line = start_line + size / (2*block_size) - 1;

    total_doc_size += size;
    total_doc_size_no_padding += unpadded_size;
    h_docSizes[i] = unpadded_size;
    h_profileScore[i] = -1;
    h_docInfo[i] = (start_line << 32) | end_line;
  }
  h_isWordInProfileHash.reset( (1L << BLOOM_SIZE) );
  h_docWordFrequencies_dimm1.reset( total_doc_size/2 );
  h_docWordFrequencies_dimm2.reset( total_doc_size/2 );

  printf("Creating Documents total_terms=%d (no_pad=%d)\n", total_doc_size, total_doc_size_no_padding);

  for (uint i=0; i<total_doc_size/2; i++) {
    h_docWordFrequencies_dimm1[i] = docEndingTag;
    h_docWordFrequencies_dimm2[i] = docEndingTag;
  }
  for (uint doci=0; doci < total_num_docs; doci++)
  {
    uint start = h_startingDocID[doci];
    uint size = h_docSizes[doci];

    for (uint i = 0; i < size/2; i++)
    {
      uint term = (rand_desh()%((1L << 24)-1));
      uint freq = (rand_desh()%254)+1;
      h_docWordFrequencies_dimm1[start + i] = (term << 8) | freq;

      term = (rand_desh()%((1L << 24)-1));
      freq = (rand_desh()%254)+1;
      h_docWordFrequencies_dimm2[start + i] = (term << 8) | freq;
    }
    if (size%2) {
      uint term = (rand_desh()%((1L << 24)-1));
      uint freq = (rand_desh()%254)+1;
      h_docWordFrequencies_dimm1[start + size/2] = (term << 8) | freq;
    }
  }

  h_profileWeights.reset( (1L << 24) );
  for (uint i=0; i<(1L << BLOOM_SIZE); i++) {
    h_isWordInProfileHash[i] = 0x0;
  }
  printf("Creating Profile\n");
  for (uint i=0; i<(1L << 24); i++) {
    h_profileWeights[i] = 0;
  }

  for (uint i=0; i<16384; i++) {
    uint entry = (rand_desh()%(1<<24));	

    h_profileWeights[entry] = 10;

    uint hash1 = entry >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
    h_isWordInProfileHash[ hash1 >> 5 ] |= 1 << (hash1 & 0x1f);
    uint hash2 = entry & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
    h_isWordInProfileHash[ hash2 >> 5 ] |= 1 << (hash2 & 0x1f);
  }
}

void runOnCPU()
{
  // go through each document in turn, and compute the score
  scoped_aligned_ptr<ulong> cpu_profileScore;
  cpu_profileScore.reset( total_num_docs );
  uint total = 0;
  uint falsies = 0;
  for (uint doci=0; doci < total_num_docs; doci++)
  {

    cpu_profileScore[doci] = 0.0;
    uint start = h_startingDocID[doci];
    uint size = h_docSizes[doci];

    for (uint i = 0; i < size/2 + (size%2); i++)
    {
      uint curr_entry = h_docWordFrequencies_dimm1[start + i];
      uint frequency = curr_entry & 0x00ff;
      uint word_id = curr_entry >> 8;
      uint hash1 = word_id >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
      bool inh1 = h_isWordInProfileHash[ hash1 >> 5 ] & ( 1 << (hash1 & 0x1f));
      uint hash2 = word_id & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
      bool inh2 = h_isWordInProfileHash[ hash2 >> 5 ] & ( 1 << (hash2 & 0x1f));

      if (inh1 && inh2)
      {
        total++;
        if (h_profileWeights[word_id] == 0) falsies++;
        cpu_profileScore[doci] += h_profileWeights[word_id] * (ulong)frequency;
      }
    }

    for (uint i = 0; i < size/2; i++)
    {
      uint curr_entry = h_docWordFrequencies_dimm2[start + i];
      uint frequency = curr_entry & 0x00ff;
      uint word_id = curr_entry >> 8;
      uint hash1 = word_id >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
      bool inh1 = h_isWordInProfileHash[ hash1 >> 5 ] & ( 1 << (hash1 & 0x1f));
      uint hash2 = word_id & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
      bool inh2 = h_isWordInProfileHash[ hash2 >> 5 ] & ( 1 << (hash2 & 0x1f));

      if (inh1 && inh2)
      {
        total++;
        if (h_profileWeights[word_id] == 0) falsies++;
        cpu_profileScore[doci] += h_profileWeights[word_id] * (ulong)frequency;
      }
    }
  }

  printf( "total_access = %d , falsies = %d, percentage = %f hit= %g\n", \
      total, falsies, total * 1.0f / total_doc_size, (total-falsies)*1.0f/total_doc_size );
  // compare the final scores
  for (uint doci = 0; doci < total_num_docs; doci++)
  {
    if (cpu_profileScore[doci] != h_profileScore[doci]) {
      printf("FAILED\n   : doc[%d] score: CPU = %lu, Device = %lu\n", \
          doci, cpu_profileScore[doci], h_profileScore[doci]);
      return;
    }
  }
  printf( "Verification: PASS\n" );
}

int main(int argc, char** argv)
{
  iris::Platform platform; if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  bool gpu=false;
  for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }
  if(!gpu) return 1;
  Options options(argc, argv);
  // Optional argument to specify the problem size.
  if(options.has("n")) {
    total_num_docs = options.get<uint>("n");
  }
  printf("Total number of documents: %u\n", total_num_docs);

  if(options.has("p")) {
    repeat = options.get<uint>("p");
  }
  printf("Kernel execution count: %u\n", repeat);

  srand(2);
  printf("RAND_MAX: %d\n", RAND_MAX);
  printf("Allocating and setting up data\n");
  setupData();

  size_t local_size = (block_size / MANUAL_VECTOR); 
  size_t global_size = total_doc_size / 2 / MANUAL_VECTOR / local_size;
  size_t local_size_reduction = block_size;
  size_t global_size_reduction = total_num_docs / block_size;

  iris::DMem d_docWordFrequencies_dimm1(h_docWordFrequencies_dimm1.get(),sizeof(uint)*total_doc_size/2);
  iris::DMem d_docWordFrequencies_dimm2(h_docWordFrequencies_dimm2.get(),sizeof(uint)*total_doc_size/2);
  std::vector<uint> partial1(total_doc_size/(2*block_size)),partial2(partial1.size());
  iris::DMem d_partialSums_dimm1(partial1.data(),partial1.size()*sizeof(uint)),d_partialSums_dimm2(partial2.data(),partial2.size()*sizeof(uint));
  iris::DMem d_profileWeights_dimm1(h_profileWeights.get(),sizeof(ulong)*(1L<<24));
  iris::DMem d_profileWeights_dimm2(h_profileWeights.get(),sizeof(ulong)*(1L<<24));
  iris::DMem d_isWordInProfileHash(h_isWordInProfileHash.get(),sizeof(uint)*(1L<<BLOOM_SIZE));
  iris::DMem d_docInfo(h_docInfo.get(),sizeof(ulong)*total_num_docs);
  iris::DMem d_profileScore(h_profileScore.get(),sizeof(ulong)*total_num_docs);

  const double start_time = getCurrentTimestamp();

  for (uint i=0; i<repeat; i++) {
    {
      iris::Task task; size_t gws[]={global_size*local_size},lws[]={local_size};
      void* args[]={&d_docWordFrequencies_dimm1,&d_docWordFrequencies_dimm2,&d_profileWeights_dimm1,&d_profileWeights_dimm2,&d_isWordInProfileHash,&d_partialSums_dimm1,&d_partialSums_dimm2};
      int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,iris_w};
      if(task.kernel("compute",1,nullptr,gws,lws,7,args,info)!=IRIS_SUCCESS) return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }

    {
      iris::Task task; size_t gws[]={global_size_reduction*local_size_reduction},lws[]={local_size_reduction};
      void* args[]={&d_docInfo,&d_partialSums_dimm1,&d_partialSums_dimm2,&d_profileScore};
      int info[]={iris_r,iris_r,iris_r,iris_w};
      if(task.kernel("reduction",1,nullptr,gws,lws,4,args,info)!=IRIS_SUCCESS) return 2;
      if(i==repeat-1) task.flush_out(d_profileScore);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 2;
    }
  }


  const double end_time = getCurrentTimestamp();
  double kernelExecutionTime = (end_time - start_time)/repeat;
  printf("======================================================\n");
  printf("Kernel Time = %f ms (averaged over %d times)\n", kernelExecutionTime * 1000.0f, repeat );
  printf("Throughput = %f\n", total_doc_size_no_padding / kernelExecutionTime / 1.0e+6f );

  if(const char* path=getenv("HECBENCH_ARTIFACT")) {
    FILE* out=fopen(path,"wb");if(!out) return 3;
    fwrite(h_profileScore.get(),sizeof(ulong),total_num_docs,out);fclose(out);
  }









  printf("Done\n");

  runOnCPU();
  int errors=platform.error_count();platform.finalize();return errors ? 1 : 0;
}
