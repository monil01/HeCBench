#include <hip/hip_runtime.h>
#include <cmath>
#include "type.h"
extern "C" __global__ void RL_s32(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wid = threadIdx.x / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  __shared__ REAL s[32];

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
  __syncthreads();

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = threadIdx.x % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += blockDim.x/32;
  }

  __syncthreads();
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_perturb_s32(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos,
    const float pert)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wid = threadIdx.x / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  __shared__ REAL s[32];

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
  __syncthreads();

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = threadIdx.x % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += blockDim.x/32;
  }

  __syncthreads();
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_s4(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wid = threadIdx.x / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  __shared__ REAL s[4];

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
  __syncthreads();

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = threadIdx.x % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += blockDim.x/32;
  }

  __syncthreads();
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_perturb_s4(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos,
    const float pert)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wid = threadIdx.x / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  __shared__ REAL s[4];

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
  __syncthreads();

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = threadIdx.x % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += blockDim.x/32;
  }

  __syncthreads();
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_s2(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wid = threadIdx.x / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  __shared__ REAL s[2];

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
  __syncthreads();

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = threadIdx.x % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += blockDim.x/32;
  }

  __syncthreads();
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_perturb_s2(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned* __restrict__ csr_r_ptr_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const int* __restrict__ level_idx_dev,
    REAL* __restrict__ tmpMem,
    const unsigned n,
    const int levelHead,
    const int inLevPos,
    const float pert)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int wid = threadIdx.x / 32;

  const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  __shared__ REAL s[2];

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
  __syncthreads();

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
  const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
  unsigned subCol;
  const int tidInWarp = threadIdx.x % 32;
  unsigned subColElem = 0;

  int woffset = 0;
  while (subMatSize > woffset)
  {
    if (wid + woffset < subMatSize)
    {
      offset = 0;
      subCol = csr_c_idx_dev[subColPos + woffset];
      while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
      {
        if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {

          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          unsigned ridx = sym_r_idx_dev[subColElem];

          if (ridx == currentCol)
          {
            s[wid] = val_dev[subColElem];
          }
          //Threads in a warp are always synchronized
          //__syncthreads();
          if (ridx > currentCol)
          {
            //elem in currentCol same row with subColElem might be 0, so
            //clearing tmpMem is necessary
            atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
          }
        }
        offset += 32;
      }
    }
    woffset += blockDim.x/32;
  }

  __syncthreads();
  //Clear tmpMem
  offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[bid*n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_onecol_factorizeCurrentCol(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  const int tid = threadIdx.x;

  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_onecol_factorizeCurrentCol_perturb(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n,
    const float pert)
{
  const int tid = threadIdx.x;

  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  //update current col

  int offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];

      if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;

      val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
      tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_onecol_updateSubmat(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    REAL* __restrict__ val_dev,
    const unsigned* __restrict__ csr_c_idx_dev,
    const unsigned* __restrict__ csr_diag_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ REAL s;

  //broadcast to submatrix
  const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
  unsigned subCol;
  unsigned subColElem = 0;

  int offset = 0;
  subCol = csr_c_idx_dev[subColPos];
  while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
  {
    if (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
      subColElem = sym_c_ptr_dev[subCol] + tid + offset;
      unsigned ridx = sym_r_idx_dev[subColElem];

      if (ridx == currentCol)
      {
        s = val_dev[subColElem];
      }
      __syncthreads();
      if (ridx > currentCol)
      {
        atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
      }
    }
    offset += blockDim.x;
  }
}

extern "C" __global__ void RL_onecol_cleartmpMem(
    const unsigned* __restrict__ sym_c_ptr_dev,
    const unsigned* __restrict__ sym_r_idx_dev,
    const unsigned* __restrict__ l_col_ptr_dev,
    const unsigned currentCol,
    REAL* __restrict__ tmpMem,
    const int stream,
    const unsigned n)
{
  const int tid = threadIdx.x;

  const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
  const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

  unsigned offset = 0;
  while (currentLColSize > offset)
  {
    if (tid + offset < currentLColSize)
    {
      unsigned ridx = sym_r_idx_dev[currentLPos + offset];
      tmpMem[stream * n + ridx]= 0;
    }
    offset += blockDim.x;
  }
}
