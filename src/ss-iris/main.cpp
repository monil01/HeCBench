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

#include "StringSearch.h"

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
      memcpy((result + count), (result + (i * searchLenPerWG)),
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

int main(int argc, char* argv[])
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

  iris::Platform platform;check(platform.init(&argc,&argv,true));int device=hip_device();
  std::vector<uchar> lower_pattern(subStrLength);
  for(uint i=0;i<subStrLength;++i)lower_pattern[i]=TOLOWER(static_cast<uchar>(subStr[i]));
  iris::DMem textBuf(text,textLength),subStrBuf(lower_pattern.data(),subStrLength);
  uint totalSearchPos = textLength - subStrLength + 1;
  uint searchLenPerWG = SEARCH_BYTES_PER_WORKITEM * LOCAL_SIZE;
  uint workGroupCount = (totalSearchPos + searchLenPerWG - 1) / searchLenPerWG;

  uint* resultCount = (uint*) malloc(workGroupCount * sizeof(uint));
  uint* result = (uint*) malloc((textLength - subStrLength + 1) * sizeof(uint));

  iris::DMem resultCountBuf(resultCount,workGroupCount*sizeof(uint));
  iris::DMem resultBuf(result,totalSearchPos*sizeof(uint));
  size_t gws=size_t(workGroupCount)*LOCAL_SIZE,lws=LOCAL_SIZE;
  int verification_status=0;
  auto launch=[&](const char* name){
    void* args[]={&textBuf,&textLength,&subStrBuf,&subStrLength,&resultBuf,&resultCountBuf,&searchLenPerWG};
    int info[]={iris_r,sizeof(textLength),iris_r,sizeof(subStrLength),iris_w,iris_w,sizeof(searchLenPerWG)};
    iris::Task task;check(task.kernel(name,1,nullptr,&gws,&lws,7,args,info));check(task.submit(device,nullptr,true));
  };
  double time = 0.0;

  if(subStrLength == 1)
  {
    std::cout <<
      "\nRun only Naive-Kernel version of String Search for pattern size = 1" <<
      std::endl;
    std::cout << "\nExecuting String search naive for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
      launch("StringSearchNaive");

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Read Results Count per workGroup
    iris::Task flush;check(flush.flush_out(resultCountBuf));check(flush.flush_out(resultBuf));check(flush.submit(device,nullptr,true));

    verification_status |= verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
  }

  if(subStrLength > 1)
  {
    std::cout << "\nExecuting String search with load balance for " <<
      iterations << " iterations" << std::endl;

    auto start = std::chrono::steady_clock::now();

    for(int i = 0; i < iterations; i++)
      launch("StringSearchLoadBalance");

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Read Results Count per workGroup
    iris::Task flush;check(flush.flush_out(resultCountBuf));check(flush.flush_out(resultBuf));check(flush.submit(device,nullptr,true));

    verification_status |= verify(resultCount, workGroupCount, result, searchLenPerWG, cpuResults);
  }

  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / iterations);

  free(text);
  free(result);
  free(resultCount);
  check(platform.finalize());
  return verification_status==0 ? 0 : 1;
}
