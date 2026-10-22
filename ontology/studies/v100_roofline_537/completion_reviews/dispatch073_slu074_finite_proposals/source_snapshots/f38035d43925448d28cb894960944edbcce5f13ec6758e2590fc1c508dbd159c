#include "study_interface.h"
#include <chrono>
#include <iostream>
#include <cmath>
#include <cuda.h>
#include "symbolic.h"
#include "Timer.h"

using namespace std;

#define TMPMEMNUM  10353
#define  Nstreams  16

__global__ void RL(
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

  extern __shared__ REAL s[];

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
        const bool active = tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol];
        unsigned ridx = 0;
        if (active) {
          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          ridx = sym_r_idx_dev[subColElem];
          if (ridx == currentCol) s[wid] = val_dev[subColElem];
        }
        // All32 lanes of this source warp participate, including partial tails.
        __syncwarp();
        if (active && ridx > currentCol)
          atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
        // Finish readers before a later warp iteration may publish another value.
        __syncwarp();
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

__global__ void RL_perturb(
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

  extern __shared__ REAL s[];

  if (tid == 0 && currentLColSize > 0 && abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
    val_dev[l_col_ptr_dev[currentCol]] = pert;
  __syncthreads();

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
        const bool active = tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol];
        unsigned ridx = 0;
        if (active) {
          subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
          ridx = sym_r_idx_dev[subColElem];
          if (ridx == currentCol) s[wid] = val_dev[subColElem];
        }
        // All32 lanes of this source warp participate, including partial tails.
        __syncwarp();
        if (active && ridx > currentCol)
          atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
        // Finish readers before a later warp iteration may publish another value.
        __syncwarp();
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

__global__ void RL_onecol_factorizeCurrentCol(
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

__global__ void RL_onecol_factorizeCurrentCol_perturb(
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

  if (tid == 0 && currentLColSize > 0 && abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
    val_dev[l_col_ptr_dev[currentCol]] = pert;
  __syncthreads();

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

__global__ void RL_onecol_updateSubmat(
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
    const bool active = tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol];
    unsigned ridx = 0;
    if (active) {
      subColElem = sym_c_ptr_dev[subCol] + tid + offset;
      ridx = sym_r_idx_dev[subColElem];
      if (ridx == currentCol) s = val_dev[subColElem];
    }
    __syncthreads();
    if (active && ridx > currentCol)
      atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
    __syncthreads();
    offset += blockDim.x;
  }
}

__global__ void RL_onecol_cleartmpMem(
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

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
  unsigned n = A_sym.n;
  unsigned nnz = A_sym.nnz;
  unsigned num_lev = A_sym.num_lev;
  unsigned *sym_c_ptr_dev, *sym_r_idx_dev, *l_col_ptr_dev;
  REAL *val_dev, *tmpMem;
  unsigned *csr_r_ptr_dev, *csr_c_idx_dev, *csr_diag_ptr_dev;
  int *level_idx_dev;

  SLU_CUDA_CHECK(cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&val_dev, nnz * sizeof(REAL)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned)));
  SLU_CUDA_CHECK(cudaMalloc((void**)&level_idx_dev, n * sizeof(int)));

  SLU_CUDA_CHECK(cudaMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice));
  SLU_CUDA_CHECK(cudaMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), cudaMemcpyHostToDevice));

  SLU_CUDA_CHECK(cudaMalloc((void**)&tmpMem, TMPMEMNUM*n*sizeof(REAL)));
  SLU_CUDA_CHECK(cudaMemset(tmpMem, 0, TMPMEMNUM*n*sizeof(REAL)));

  // calculate 1-norm of A and perturbation value for perturbation
  float pert = 0;
  if (PERTURB)
  {
    float norm_A = 0;
    for (unsigned i = 0; i < n; ++i)
    {
      float tmp = 0;
      for (unsigned j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
        tmp += abs(A_sym.val[j]);
      if (norm_A < tmp)
        norm_A = tmp;
    }
    pert = 3.45e-4 * norm_A;
    out << "Gaussian elimination with static pivoting (GESP)..." << endl;
    out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
  }

  SLU_CUDA_CHECK(cudaDeviceSynchronize());

  void* study_pointers[]={sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem};
  study_before(A_sym,study_pointers,tmpMem,PERTURB,pert);
  SLU_CUDA_CHECK(cudaDeviceSynchronize());
  auto study_start=std::chrono::steady_clock::now();
  Timer t;
  double utime;
  t.start();
  for (unsigned i = 0; i < num_lev; ++i)
  {
    int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];

    if (lev_size > 896) { //3584 / 4
      unsigned WarpsPerBlock = 2;
      dim3 dimBlock(WarpsPerBlock * 32, 1);
      size_t MemSize = WarpsPerBlock * sizeof(REAL);

      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        dim3 dimGrid(restCol, 1);
        if (!PERTURB)
          { SCall q={0,int(dimBlock.x),int(dimGrid.x),int(MemSize),n,nnz,0,A_sym.level_ptr[i],j*TMPMEMNUM,0,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[0]++;if(study_cb){if(study_cb(&q))abort();}else RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
              sym_r_idx_dev,
              val_dev,
              l_col_ptr_dev,
              csr_r_ptr_dev,
              csr_c_idx_dev,
              csr_diag_ptr_dev,
              level_idx_dev,
              tmpMem,
              n,
              A_sym.level_ptr[i],
              j*TMPMEMNUM); SLU_CUDA_CHECK(cudaGetLastError()); }
        else
          { SCall q={1,int(dimBlock.x),int(dimGrid.x),int(MemSize),n,nnz,0,A_sym.level_ptr[i],j*TMPMEMNUM,0,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[1]++;if(study_cb){if(study_cb(&q))abort();}else RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
              sym_r_idx_dev,
              val_dev,
              l_col_ptr_dev,
              csr_r_ptr_dev,
              csr_c_idx_dev,
              csr_diag_ptr_dev,
              level_idx_dev,
              tmpMem,
              n,
              A_sym.level_ptr[i],
              j*TMPMEMNUM,
              pert); SLU_CUDA_CHECK(cudaGetLastError()); }
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > 448) {
      unsigned WarpsPerBlock = 4;
      dim3 dimBlock(WarpsPerBlock * 32, 1);
      size_t MemSize = WarpsPerBlock * sizeof(REAL);

      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        dim3 dimGrid(restCol, 1);
        if (!PERTURB)
          { SCall q={0,int(dimBlock.x),int(dimGrid.x),int(MemSize),n,nnz,0,A_sym.level_ptr[i],j*TMPMEMNUM,0,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[0]++;if(study_cb){if(study_cb(&q))abort();}else RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
              sym_r_idx_dev,
              val_dev,
              l_col_ptr_dev,
              csr_r_ptr_dev,
              csr_c_idx_dev,
              csr_diag_ptr_dev,
              level_idx_dev,
              tmpMem,
              n,
              A_sym.level_ptr[i],
              j*TMPMEMNUM); SLU_CUDA_CHECK(cudaGetLastError()); }
        else
          { SCall q={1,int(dimBlock.x),int(dimGrid.x),int(MemSize),n,nnz,0,A_sym.level_ptr[i],j*TMPMEMNUM,0,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[1]++;if(study_cb){if(study_cb(&q))abort();}else RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
              sym_r_idx_dev,
              val_dev,
              l_col_ptr_dev,
              csr_r_ptr_dev,
              csr_c_idx_dev,
              csr_diag_ptr_dev,
              level_idx_dev,
              tmpMem,
              n,
              A_sym.level_ptr[i],
              j*TMPMEMNUM,
              pert); SLU_CUDA_CHECK(cudaGetLastError()); }
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else if (lev_size > Nstreams) {
      dim3 dimBlock(256, 1);
      size_t MemSize = 32 * sizeof(REAL);
      unsigned j = 0;
      while(lev_size > 0) {
        unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
        dim3 dimGrid(restCol, 1);
        if (!PERTURB)
          { SCall q={0,int(dimBlock.x),int(dimGrid.x),int(MemSize),n,nnz,0,A_sym.level_ptr[i],j*TMPMEMNUM,0,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[0]++;if(study_cb){if(study_cb(&q))abort();}else RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
              sym_r_idx_dev,
              val_dev,
              l_col_ptr_dev,
              csr_r_ptr_dev,
              csr_c_idx_dev,
              csr_diag_ptr_dev,
              level_idx_dev,
              tmpMem,
              n,
              A_sym.level_ptr[i],
              j*TMPMEMNUM); SLU_CUDA_CHECK(cudaGetLastError()); }
        else
          { SCall q={1,int(dimBlock.x),int(dimGrid.x),int(MemSize),n,nnz,0,A_sym.level_ptr[i],j*TMPMEMNUM,0,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[1]++;if(study_cb){if(study_cb(&q))abort();}else RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
              sym_r_idx_dev,
              val_dev,
              l_col_ptr_dev,
              csr_r_ptr_dev,
              csr_c_idx_dev,
              csr_diag_ptr_dev,
              level_idx_dev,
              tmpMem,
              n,
              A_sym.level_ptr[i],
              j*TMPMEMNUM,
              pert); SLU_CUDA_CHECK(cudaGetLastError()); }
        j++;
        lev_size -= TMPMEMNUM;
      }
    }
    else { // "Big" levels
      for (int offset = 0; offset < lev_size; offset += Nstreams) {
        for (int j = 0; j < Nstreams; j++) {
          if (j + offset < lev_size) {
            const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
            const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
              - A_sym.csr_diag_ptr[currentCol] - 1;

            if (!PERTURB)
              { SCall q={2,256,1,0,n,nnz,currentCol,0,0,j,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[2]++;if(study_cb){if(study_cb(&q))abort();}else RL_onecol_factorizeCurrentCol<<<1, 256>>>(sym_c_ptr_dev,
                  sym_r_idx_dev,
                  val_dev,
                  l_col_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n); SLU_CUDA_CHECK(cudaGetLastError()); }
            else
              { SCall q={3,256,1,0,n,nnz,currentCol,0,0,j,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[3]++;if(study_cb){if(study_cb(&q))abort();}else RL_onecol_factorizeCurrentCol_perturb<<<1, 256>>>(sym_c_ptr_dev,
                  sym_r_idx_dev,
                  val_dev,
                  l_col_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n,
                  pert); SLU_CUDA_CHECK(cudaGetLastError()); }
            if (subMatSize > 0)
              { SCall q={4,256,int(subMatSize),0,n,nnz,currentCol,0,0,j,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[4]++;if(study_cb){if(study_cb(&q))abort();}else RL_onecol_updateSubmat<<<subMatSize, 256>>>(sym_c_ptr_dev,
                  sym_r_idx_dev,
                  val_dev,
                  csr_c_idx_dev,
                  csr_diag_ptr_dev,
                  currentCol,
                  tmpMem,
                  j,
                  n); SLU_CUDA_CHECK(cudaGetLastError()); }
            { SCall q={5,256,1,0,n,nnz,currentCol,0,0,j,pert,0 ,{sym_c_ptr_dev,sym_r_idx_dev,val_dev,l_col_ptr_dev,csr_r_ptr_dev,csr_c_idx_dev,csr_diag_ptr_dev,level_idx_dev,tmpMem}};study_counts[5]++;if(study_cb){if(study_cb(&q))abort();}else RL_onecol_cleartmpMem<<<1, 256>>>(sym_c_ptr_dev,
                sym_r_idx_dev,
                l_col_ptr_dev,
                currentCol,
                tmpMem,
                j,
                n); SLU_CUDA_CHECK(cudaGetLastError()); }
          }
        }
      }
    }
  }
  SLU_CUDA_CHECK(cudaDeviceSynchronize());
  study_roi_seconds(std::chrono::duration<double>(std::chrono::steady_clock::now()-study_start).count());
  t.elapsedUserTime(utime);
  out << "Total LU kernel execution time: " << utime << " ms" << std::endl;

  //copy LU val back to main mem
  SLU_CUDA_CHECK(cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost));
  study_after(A_sym,study_pointers,tmpMem);

#ifdef VERIFY
  //check NaN elements
  unsigned err_find = 0;
  for(unsigned i = 0; i < nnz; i++)
    if(isnan(A_sym.val[i]) || isinf(A_sym.val[i])) 
      err_find++;

  if (err_find != 0)
    err << "LU data check: NaN found!!" << std::endl;
#endif

  SLU_CUDA_CHECK(cudaFree(sym_c_ptr_dev));
  SLU_CUDA_CHECK(cudaFree(sym_r_idx_dev));
  SLU_CUDA_CHECK(cudaFree(val_dev));
  SLU_CUDA_CHECK(cudaFree(l_col_ptr_dev));
  SLU_CUDA_CHECK(cudaFree(csr_c_idx_dev));
  SLU_CUDA_CHECK(cudaFree(csr_r_ptr_dev));
  SLU_CUDA_CHECK(cudaFree(csr_diag_ptr_dev));
  SLU_CUDA_CHECK(cudaFree(level_idx_dev));
  SLU_CUDA_CHECK(cudaFree(tmpMem));
}
