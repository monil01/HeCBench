/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  • Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  • Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <chrono>
#include <cuda.h>
#include "StringSearch.h"
#include "reference_history/kernels.cu"

int verify(uint* resultCount, uint workGroupCount, 
    uint* result, uint searchLenPerWG, 
    std::vector<uint> &cpuResults) 
{
  uint count = resultCount[0];
  for(uint i=1; i<workGroupCount; ++i)
  {
    uint found = resultCount[i];
    if(found > 0)
    {
      memmove((result + count), (result + (i * searchLenPerWG)),
          found * sizeof(uint));
      count += found;
    }
  }
  std::sort(result, result+count);

  std::cout << "Device: found " << count << " times\n"; 

  // compare the results and see if they match
  bool pass = (count == cpuResults.size());
  pass = pass && std::equal (result, result+count, cpuResults.begin());
  if(pass)
  {
    std::cout << "Passed!\n" << std::endl;
    return 0;
  }
  else
  {
    std::cout << "Failed\n" << std::endl;
    return -1;
  }
}

int source_main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <path to file> <substring> <repeat>\n", argv[0]);
    return -1;
  }
  std::string file = std::string(argv[1]); // "StringSearch_Input.txt";
  std::string subStr = std::string(argv[2]);
  int iterations = atoi(argv[3]);

  if(iterations < 1)
  {
    std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  // Check input text-file specified.
  if(file.length() == 0)
  {
    std::cout << "\n Error: Input File not specified..." << std::endl;
    return -1;
  }

  // Read the content of the file
  std::ifstream textFile(file.c_str(),
      std::ios::in|std::ios::binary|std::ios::ate);
  if(! textFile.is_open())
  {
    std::cout << "\n Unable to open file: " << file << std::endl;
    return -1;
  }

  uint textLength = (uint)(textFile.tellg());
  uchar* text = (uchar*)malloc(textLength+1);
  memset(text, 0, textLength+1);
  textFile.seekg(0, std::ios::beg);
  if (!textFile.read ((char*)text, textLength))
  {
    std::cout << "\n Reading file failed " << std::endl;
    textFile.close();
    return -1;
  }
  textFile.close();

  uint subStrLength = subStr.length();
  if(subStrLength == 0)
  {
    std::cout << "\nError: Sub-String not specified..." << std::endl;
    return -1;
  }

  if (textLength < subStrLength)
  {
    std::cout << "\nText size less than search pattern (" << textLength
      << " < " << subStrLength << ")" << std::endl;
    return -1;
  }

#ifdef ENABLE_2ND_LEVEL_FILTER
  if(subStrLength != 1 && subStrLength <= 16)
  {
    std::cout << "\nSearch pattern size should be longer than 16" << std::endl;
    return -1;
  }
#endif

  std::cout << "Search Pattern : " << subStr << std::endl;

  // Rreference implementation on host device
  std::vector<uint> cpuResults;

  uint last = subStrLength - 1;
  uint badCharSkip[UCHAR_MAX + 1];

  // Initialize the table with default values
  uint scan = 0;
  for(scan = 0; scan <= UCHAR_MAX; ++scan)
  {
    badCharSkip[scan] = subStrLength;
  }

  // populate the table with analysis on pattern
  for(scan = 0; scan < last; ++scan)
  {
    badCharSkip[toupper(subStr[scan])] = last - scan;
    badCharSkip[tolower(subStr[scan])] = last - scan;
  }

  // search the text
  uint curPos = 0;
  while((textLength - curPos) > last)
  {
    int p=last;
    for(scan=(last+curPos); COMPARE(text[scan], subStr[p--]); scan -= 1)
    {
      if (scan == curPos)
      {
        cpuResults.push_back(curPos);
        break;
      }
    }
    curPos += (scan == curPos) ? 1 : badCharSkip[text[last+curPos]];
  }

  std::cout << "CPU: found " << cpuResults.size() << " times\n"; 

  uchar *textBuf;
  study_allocate((void**)&textBuf,textLength);
  SS_CUDA_CHECK(cudaMemcpy(textBuf, text, textLength, cudaMemcpyHostToDevice));

  // Move subStr data host to device
  const uchar* ss = (const uchar*) subStr.c_str();
  uchar *subStrBuf;
  study_allocate((void**)&subStrBuf,subStrLength);
  SS_CUDA_CHECK(cudaMemcpy(subStrBuf, ss, subStrLength, cudaMemcpyHostToDevice));

  uint totalSearchPos = textLength - subStrLength + 1;
  uint searchLenPerWG = SEARCH_BYTES_PER_WORKITEM * LOCAL_SIZE;
  uint workGroupCount = (totalSearchPos + searchLenPerWG - 1) / searchLenPerWG;

  uint* resultCount = (uint*) malloc(workGroupCount * sizeof(uint));
  uint* result = (uint*) malloc((textLength - subStrLength + 1) * sizeof(uint));

  uint *resultCountBuf;
  study_allocate((void**)&resultCountBuf,workGroupCount * sizeof(uint));

  uint *resultBuf;
  study_allocate((void**)&resultBuf,(textLength - subStrLength + 1) * sizeof(uint));

  dim3 block (LOCAL_SIZE);
  dim3 grid (workGroupCount);

  double time = 0.0;
  study_before(textBuf,textLength,subStrBuf,subStrLength,resultBuf,resultCountBuf,workGroupCount,totalSearchPos,searchLenPerWG,text,ss);

  if(subStrLength == 1)
  {
    std::cout <<
      "\nRun only Naive-Kernel version of String Search for pattern size = 1" <<
      std::endl;
    std::cout << "\nExecuting String search naive for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
      {SSCall q={0,int(block.x),int(grid.x),int(subStrLength),textLength,subStrLength,searchLenPerWG,0,{textBuf,subStrBuf,resultBuf,resultCountBuf}};study_counts[0]++;if(study_callback){if(study_callback(&q))throw std::runtime_error("authoredframeworkcallbackfailed");}else StringSearchNaive<<<grid, block, subStrLength>>>(
          textBuf,
          textLength,
          subStrBuf, 
          subStrLength,
          resultBuf, 
          resultCountBuf, 
          searchLenPerWG); SS_CUDA_CHECK(cudaGetLastError());}

    SS_CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    study_roi_seconds(std::chrono::duration<double>(end-start).count());
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Read Results Count per workGroup
    SS_CUDA_CHECK(cudaMemcpy(resultCount, resultCountBuf, workGroupCount * sizeof(uint), cudaMemcpyDeviceToHost));
    SS_CUDA_CHECK(cudaMemcpy(result, resultBuf, (textLength - subStrLength + 1) * sizeof(uint), cudaMemcpyDeviceToHost));

    study_device_outputs();
    int study_verify_status=verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
    study_source_final(resultCount,workGroupCount,result,cpuResults,study_verify_status); 
  }

  if(subStrLength > 1) 
  {
    std::cout << "\nExecuting String search with load balance for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
      {SSCall q={1,int(block.x),int(grid.x),int(subStrLength),textLength,subStrLength,searchLenPerWG,0,{textBuf,subStrBuf,resultBuf,resultCountBuf}};study_counts[1]++;if(study_callback){if(study_callback(&q))throw std::runtime_error("authoredframeworkcallbackfailed");}else StringSearchLoadBalance<<<grid, block, subStrLength>>>(
          textBuf,
          textLength,
          subStrBuf, 
          subStrLength,
          resultBuf, 
          resultCountBuf, 
          searchLenPerWG); SS_CUDA_CHECK(cudaGetLastError());}

    SS_CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    study_roi_seconds(std::chrono::duration<double>(end-start).count());
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Read Results Count per workGroup
    SS_CUDA_CHECK(cudaMemcpy(resultCount, resultCountBuf, workGroupCount * sizeof(uint), cudaMemcpyDeviceToHost));
    SS_CUDA_CHECK(cudaMemcpy(result, resultBuf, (textLength - subStrLength + 1) * sizeof(uint), cudaMemcpyDeviceToHost));

    study_device_outputs();
    int study_verify_status=verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
    study_source_final(resultCount,workGroupCount,result,cpuResults,study_verify_status); 
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);

  study_free(resultCountBuf);
  study_free(resultBuf);
  study_free(subStrBuf);
  study_free(textBuf);
  free(text);
  free(result);
  free(resultCount);
  return 0;
}
