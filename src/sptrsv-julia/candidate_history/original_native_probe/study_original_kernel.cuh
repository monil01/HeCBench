#ifndef _SPTRSV_SYNCFREE_
#define _SPTRSV_SYNCFREE_

#include <chrono>
#include <cuda.h>
#include "sptrsv.h"

// reference
// https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda

// addr must be aligned properly.
__device__ int atomic_load(const int *addr)
{
  const volatile int *vaddr = addr; // volatile to bypass cache
  //__threadfence(); // for seq_cst loads. Remove for acquire semantics.
  const int value = *vaddr;
  // fence to ensure that dependent reads are correctly ordered
  __threadfence(); 
  return value; 
}

// addr must be aligned properly.
__device__ void atomic_store(int *addr, int value)
{
  volatile int *vaddr = addr; // volatile to bypass cache
  // fence to ensure that previous non-atomic stores are visible to other threads
  __threadfence(); 
  *vaddr = value;
}

__global__
void sptrsv_mix(
    const int        *__restrict__ csrRowPtr,
    const int        *__restrict__ csrColIdx,
    const VALUE_TYPE *__restrict__ csrVal,
    int              *__restrict__ get_value,
    const int        m,
    const VALUE_TYPE *__restrict__ b,
    VALUE_TYPE       *__restrict__ x,
    const int        *__restrict__ warp_num,
    const int        Len)
{
  const int local_id = threadIdx.x;
  const int global_id = blockIdx.x * blockDim.x + local_id;
  const int warp_id = global_id/WARP_SIZE;

  int row;
  __shared__ VALUE_TYPE s_left_sum[WARP_PER_BLOCK*WARP_SIZE];

  if(warp_id>=(Len-1)) return;

  const int lane_id = (WARP_SIZE - 1) & local_id;

  if(warp_num[warp_id+1]>(warp_num[warp_id]+1))
  {
    //thread
    row =warp_num[warp_id]+lane_id;
    if(row>=m) return;

    int col,j,i;
    VALUE_TYPE xi;
    VALUE_TYPE left_sum=0;
    i=row;
    j=csrRowPtr[i];

    while(j<csrRowPtr[i+1])
    {
      col=csrColIdx[j];
      if(atomic_load(&get_value[col])==1)
      {
        left_sum+=csrVal[j]*x[col];
        j++;
        col=csrColIdx[j];
      }
      if(i==col)
      {
        xi = (b[i] - left_sum) / csrVal[csrRowPtr[i+1]-1];
        x[i] = xi;
        atomic_store(&get_value[i], 1);
        j++;
      }
    }
  }
  else
  {
    row = warp_num[warp_id];
    if(row>=m)
      return;

    int col,j=csrRowPtr[row]  + lane_id;
    VALUE_TYPE xi,sum=0;
    while(j < (csrRowPtr[row+1]-1))
    {
      col=csrColIdx[j];
      if(atomic_load(&get_value[col])==1)
      {
        sum += x[col] * csrVal[j];
        j += WARP_SIZE;
      }
    }

    s_left_sum[local_id]=sum;

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    {
      if(lane_id < offset)
      {
        s_left_sum[local_id] += s_left_sum[local_id+offset];
      }
    }

    if (!lane_id)
    {
      xi = (b[row] - s_left_sum[local_id]) / csrVal[csrRowPtr[row+1]-1];
      x[row]=xi;
      atomic_store(&get_value[row], 1);
    }
  }
}


#endif
