#include <cuda_runtime.h>
#include "bucketsort.h"
#include "kernel_bucketcount.h"
#include "kernel_bucketprefix.h"
#include "kernel_bucketsort.h"
#include "kernel_histogram.h"
__device__
float4 sortElem(float4 r) {
  float4 nr;

  float xt = r.x;
  float yt = r.y;
  float zt = r.z;
  float wt = r.w;

  float nr_xt = xt > yt ? yt : xt;
  float nr_yt = yt > xt ? yt : xt;
  float nr_zt = zt > wt ? wt : zt;
  float nr_wt = wt > zt ? wt : zt;

  xt = nr_xt > nr_zt ? nr_zt : nr_xt;
  yt = nr_yt > nr_wt ? nr_wt : nr_yt;
  zt = nr_zt > nr_xt ? nr_zt : nr_xt;
  wt = nr_wt > nr_yt ? nr_wt : nr_yt;

  nr.x = xt;
  nr.y = yt > zt ? zt : yt;
  nr.z = zt > yt ? zt : yt;
  nr.w = wt;
  return nr;
}

  extern "C" __global__ void
sortElement(float4* result, float4* input, const int size) 
{
  int gid = blockIdx.x*blockDim.x+threadIdx.x;
  if (gid < size) result[gid] = sortElem(input[gid]);
}

  __device__
float4 getLowest(float4 a, float4 b)
{
  float ax = a.x;
  float ay = a.y;
  float az = a.z;
  float aw = a.w;
  float bx = b.x;
  float by = b.y;
  float bz = b.z;
  float bw = b.w;
  a.x = ax < bw ? ax : bw;
  a.y = ay < bz ? ay : bz;
  a.z = az < by ? az : by;
  a.w = aw < bx ? aw : bx;
  return a;
}

  __device__
float4 getHighest(float4 a, float4 b)
{
  float ax = a.x;
  float ay = a.y;
  float az = a.z;
  float aw = a.w;
  float bx = b.x;
  float by = b.y;
  float bz = b.z;
  float bw = b.w;
  b.x = aw >= bx ? aw : bx;
  b.y = az >= by ? az : by;
  b.z = ay >= bz ? ay : bz;
  b.w = ax >= bw ? ax : bw;
  return b;
}

// the kernel calls the functions defined above
#include "kernel_mergeSortPass.h"

  extern "C" __global__ void
mergepack ( float* result , 
    const float* orig , 
    const int *constStartAddr,
    const unsigned int *finalStartAddr,
    const unsigned int *nullElems )
{

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int division = blockIdx.y;
  if((finalStartAddr[division] + gid) < finalStartAddr[division + 1])
    result[finalStartAddr[division] + gid] = 
      orig[constStartAddr[division]*4 + nullElems[division] + gid];
}
