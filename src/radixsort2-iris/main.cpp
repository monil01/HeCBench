#include <random>
#include <algorithm>
#include <chrono>
#include <climits>






#include "helper_string.h"
#include "iris_support.hpp"
#include <vector>
#include <cstdint>
#include <cstring>
#include <numeric>


template <typename T, bool floatKeys>
bool testSort(int argc, char **argv) {
  int cmdVal;
  int keybits = 32;

  int numElements = 1048576;
  bool keysOnly = checkCmdLineFlag(argc, (const char **)argv, "keysonly");
  bool quiet = checkCmdLineFlag(argc, (const char **)argv, "quiet");

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    numElements = cmdVal;

    if (cmdVal < 0) {
      printf("Error: elements must be > 0, elements=%d is invalid\n", cmdVal);
      exit(EXIT_SUCCESS);
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "keybits")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "keybits");
    keybits = cmdVal;

    if (keybits <= 0) {
      printf("Error: keybits must be > 0, keybits=%d is invalid\n", keybits);
      exit(EXIT_SUCCESS);
    }
  }

  unsigned int numIterations = (numElements >= 16777216) ? 10 : 100;

  if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
    cmdVal = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    numIterations = cmdVal;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printf("Command line:\nradixSortThrust [-option]\n");
    printf("Valid options:\n");
    printf("-n=<N>        : number of elements to sort\n");
    printf("-keybits=bits : keybits must be > 0\n");
    printf(
        "-keysonly     : only sort an array of keys (default sorts key-value "
        "pairs)\n");
    printf(
        "-float        : use 32-bit float keys (default is 32-bit unsigned "
        "int)\n");
    printf(
        "-quiet        : Output only the number of elements and the time to "
        "sort\n");
    printf("-help         : Output a help message\n");
    exit(EXIT_SUCCESS);
  }

  if (!quiet)
    printf("\nSorting %d %d-bit %s keys %s\n\n", numElements, keybits,
           floatKeys ? "float" : "unsigned int",
           keysOnly ? "(only)" : "and values");

  std::vector<T> h_keys(numElements);
  std::vector<T> h_keysSorted(numElements);
  std::vector<unsigned int> h_values;

  if (!keysOnly) h_values = std::vector<unsigned int>(numElements);

  std::mt19937 rng (19937);
  if (floatKeys) {
    std::uniform_real_distribution<float> u01(0, 1);
    for (int i = 0; i < numElements; i++) h_keys[i] = u01(rng);
  } else {
    std::uniform_int_distribution<unsigned int> u(0, UINT_MAX);
    for (int i = 0; i < numElements; i++) h_keys[i] = u(rng);
  }

  if (!keysOnly) 
    for (int i = 0; i < numElements; i++) h_values[i] = i;

  // Copy data onto the GPU
  iris::Platform platform;
  int device=init_iris(platform,argc,argv);
  std::vector<T> workingKeys(numElements),scratchKeys(numElements);
  std::vector<unsigned int> workingValues(keysOnly?1:numElements),scratchValues(keysOnly?1:numElements);
  std::vector<int> positions(size_t(numElements)+1),sums((numElements+1+255)/256);
  // Empty input has no device computation, as in native Thrust.
  if(numElements==0) {
    printf("Throughput = 0.0000 MElements/s, Time = 0.00000 s\n");
    check_iris(platform.finalize());return true;
  }
  iris::DMem frontKeys(workingKeys.data(),numElements*sizeof(T));
  iris::DMem backKeys(scratchKeys.data(),numElements*sizeof(T));
  iris::DMem frontValues(workingValues.data(),workingValues.size()*sizeof(unsigned int));
  iris::DMem backValues(scratchValues.data(),scratchValues.size()*sizeof(unsigned int));
  iris::DMem prefix(positions.data(),positions.size()*sizeof(int));
  iris::DMem block_sums(sums.data(),sums.size()*sizeof(int));
  int value_mode=keysOnly?1:0;
  double totalTime = 0;

  for (unsigned int i = 0; i < numIterations; i++) {
    // reset data before sort
    std::copy(h_keys.begin(),h_keys.end(),workingKeys.begin());
    frontKeys.update(workingKeys.data());
    if(!keysOnly) {
      std::copy(h_values.begin(),h_values.end(),workingValues.begin());
      frontValues.update(workingValues.data());
    }


    auto start = std::chrono::steady_clock::now();

    iris::DMem* inputKeys=&frontKeys;iris::DMem* outputKeys=&backKeys;
    iris::DMem* inputValues=&frontValues;iris::DMem* outputValues=&backValues;
    int scan_size=numElements+1,blocks=sums.size();
    const char* flag_name=floatKeys?"flags_float":"flags_uint";
    const char* scatter_name=floatKeys?"scatter_float":"scatter_uint";
    for(int bit=0;bit<32;++bit) {
      {
        void* p[]={ inputKeys,&prefix,&numElements,&bit };
        int info[]={ iris_r,iris_w,sizeof(numElements),sizeof(bit) };
        launch_iris(device,flag_name,((size_t(scan_size)+255)/256)*256,256,4,p,info);
      }
      {
        void* p[]={ &prefix,&block_sums,&scan_size };
        int info[]={ iris_rw,iris_w,sizeof(scan_size) };
        launch_iris(device,"scan_blocks",((size_t(scan_size)+255)/256)*256,256,3,p,info);
        void* p2[]={ &block_sums,&blocks };int i2[]={ iris_rw,sizeof(blocks) };
        launch_iris(device,"scan_sums",256,256,2,p2,i2);
        void* p3[]={ &prefix,&block_sums,&scan_size };int i3[]={ iris_rw,iris_r,sizeof(scan_size) };
        launch_iris(device,"scan_add",((size_t(scan_size)+255)/256)*256,256,3,p3,i3);
      }
      {
        void* p[]={ inputKeys,inputValues,outputKeys,outputValues,&prefix,&numElements,&bit,&value_mode };
        int info[]={ iris_r,iris_r,iris_w,iris_w,iris_r,sizeof(numElements),sizeof(bit),sizeof(value_mode) };
        launch_iris(device,scatter_name,((size_t(numElements)+255)/256)*256,256,8,p,info);
      }
      std::swap(inputKeys,outputKeys);std::swap(inputValues,outputValues);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    totalTime += time;
  }

  totalTime = totalTime * 1e-9f / numIterations;

  printf("Throughput = %.4f MElements/s, Time = %.5lf s\n",
         1.0e-6f * numElements / totalTime, totalTime);

  // Get results back to host for correctness checking
  if(numIterations) {
    flush_iris(device,frontKeys);
    h_keysSorted=workingKeys;
    if(!keysOnly) {flush_iris(device,frontValues);h_values=workingValues;}
  } else {h_keysSorted.clear();h_values.clear();}

  // Check results
  if(const char* path=getenv("RADIX_VALIDATION_DUMP")) {
    FILE* fp=fopen(path,"wb");if(!fp)return false;
    uint32_t header[]={uint32_t(h_keysSorted.size()),uint32_t(keysOnly),uint32_t(floatKeys)};
    fwrite(header,sizeof(uint32_t),3,fp);
    fwrite(h_keysSorted.data(),sizeof(T),h_keysSorted.size(),fp);
    if(!keysOnly) fwrite(h_values.data(),sizeof(unsigned int),h_values.size(),fp);
    fclose(fp);
  }
  bool ok=std::is_sorted(h_keysSorted.begin(),h_keysSorted.end()) && !platform.error_count();
  check_iris(platform.finalize());return ok;
}

int main(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);

  bool bTestResult = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "float"))
    bTestResult = testSort<float, true>(argc, argv);
  else
    bTestResult = testSort<unsigned int, false>(argc, argv);

  printf(bTestResult ? "PASS\n" : "FAIL\n");
}
