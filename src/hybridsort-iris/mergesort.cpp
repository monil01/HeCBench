#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include "mergesort.h"
#include "dmem.h"

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE  256
#define ROW_LENGTH  BLOCKSIZE * 4
#define ROWS    4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////


float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets){

  int *startaddr = (int *)malloc((divisions + 1)*sizeof(int));
  int largestSize = -1;
  startaddr[0] = 0;
  for(int i=1; i<=divisions; i++)
  {
    startaddr[i] = startaddr[i-1] + sizes[i-1];
    if(sizes[i-1] > largestSize) largestSize = sizes[i-1];
  }
  largestSize *= 4;

#ifdef MERGE_WG_SIZE_0
  const int THREADS = MERGE_WG_SIZE_0;
#else
  const int THREADS = 256;
#endif
  size_t local[] = {THREADS,1,1};
  size_t blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1;
  size_t grid[3];

  // divided by four 
  void* d_resultList_buff;
  allocDeviceBuffer(&d_resultList_buff, sizeof(float4)*listsize/4);

  void* d_origList_buff;
  allocDeviceBuffer(&d_origList_buff, sizeof(float4)*listsize/4);
  copyToDevice(d_origList_buff, d_origList, sizeof(float4)*listsize/4);

  void* d_constStartAddr;
  allocDeviceBuffer(&d_constStartAddr, sizeof(int)*(divisions+1));
  copyToDevice(d_constStartAddr, startaddr, sizeof(int)*(divisions+1));

  launch("sortElement", blocks, THREADS, {memory(d_resultList_buff), memory(d_origList_buff), scalar(listsize/4)});

  int nrElems = 2;

  while(true){
    int floatsperthread = (nrElems*4);
    int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread);
    int threadsNeeded = threadsPerDiv * divisions;

#ifdef MERGE_WG_SIZE_1
    local[0] = MERGE_WG_SIZE_1;
#else
    local[0] = 208;
#endif

    grid[0] = ((threadsNeeded%local[0]) == 0) ?
      threadsNeeded/local[0] :
      (threadsNeeded/local[0]) + 1;
    if(grid[0] < 8){
      grid[0] = 8;
      local[0] = ((threadsNeeded%grid[0]) == 0) ?
        threadsNeeded / grid[0] :
        (threadsNeeded / grid[0]) + 1;
    }
    // Swap orig/result list
    void* tempList = d_origList_buff;
    d_origList_buff = d_resultList_buff;
    d_resultList_buff = tempList;

    launch("mergeSortPass", grid[0], local[0], {memory(d_origList_buff), memory(d_resultList_buff), memory(d_constStartAddr), scalar(threadsPerDiv), scalar(nrElems), scalar(listsize/4)});

    nrElems *= 2;
    floatsperthread = (nrElems*4);

    if(threadsPerDiv == 1) break;
  }


#ifdef MERGE_WG_SIZE_0
  local[0] = MERGE_WG_SIZE_0;
#else
  local[0] = 256;
#endif
  grid[0] = ((largestSize%local[0]) == 0) ?  largestSize/local[0] : (largestSize/local[0]) + 1;
  grid[1] = divisions;

  void* d_finalStartAddr;
  allocDeviceBuffer(&d_finalStartAddr, sizeof(unsigned int)*(divisions+1));
  copyToDevice(d_finalStartAddr, origOffsets,  sizeof(unsigned int)*(divisions+1));

  void* d_nullElements;
  allocDeviceBuffer(&d_nullElements, sizeof(unsigned int)*divisions);
  copyToDevice(d_nullElements, nullElements, sizeof(unsigned int)*divisions);

  dim3 grids(grid[0], grid[1]); 
  dim3 threads(local[0], local[1]); 

  launch2("mergepack", grids, threads, {memory(d_origList_buff), memory(d_resultList_buff), memory(d_constStartAddr), memory(d_finalStartAddr), memory(d_nullElements)});

  copyFromDevice(d_origList, d_origList_buff, sizeof(float4)*listsize/4);

  freeDeviceBuffer(d_resultList_buff);
  freeDeviceBuffer(d_origList_buff);
  freeDeviceBuffer(d_finalStartAddr);
  freeDeviceBuffer(d_constStartAddr);
  freeDeviceBuffer(d_nullElements);
  free(startaddr);
  return d_origList;
}
