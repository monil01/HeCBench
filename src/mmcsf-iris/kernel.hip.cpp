#include <hip/hip_runtime.h>
#include <cstddef>
#define DTYPE float
#define ITYPE size_t
// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW) {

  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = (gId >> (5 + logOfWPC)) << logOfFPW; // 5: minimum 1 WARP (2^5) // blockIdx.x ;//

  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){

    tmp_val = 0;
    bool diffFiber = false;
    unsigned int idx0;

    for (int fr = 0; fr < fbrPerWarp && (fbr+fr) < (nFibers - 1); ++fr){

      diffFiber = false;
      unsigned int idx1 = fbrIdx1[fbr+fr];// dInds1[fbrPtr1[fbr]];
      idx0 = fbrLikeSlcInds[fbr+fr];//slc;
      tmp_val = 0;

      for(unsigned int x = fbrPtr1[fbr+fr] + workId; x < fbrPtr1[fbr+fr+1]; x+=warpPerSlice) {

        unsigned int idx2 = dInds2[x];

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp_val += vals[x] * dU2[idx2 * R + r]; //2MR
        }
      }

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp += tmp_val * dU1[idx1 * R + r] ; //2PR
      }

      if(fbrLikeSlcInds[fbr+fr] != fbrLikeSlcInds[fbr+fr+1]) {

        diffFiber = true;
        for(unsigned int r=laneId; r<R; r+=32) {
          atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
        }
        tmp = 0;
      }
    }

    if(!diffFiber) {
      for(unsigned int r=laneId; r<R; r+=32) {
        atomicAdd(&dU0[idx0 * R + r], tmp);
      }
    }
  }
}

// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW) {
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = (gId >> (5 + logOfWPC)) << logOfFPW; // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val, tmp2= 0;

  if(fbrS < nFibers - 1){

    tmp_val = 0;
    bool diffFiber = false;
    unsigned int idx0;

    for (int fr = 0; fr < fbrPerWarp && (fbrS+fr) < (nFibers - 1); ++fr){

      diffFiber = false;
      unsigned int idx1 = fbrIdx1[fbrS+fr];// dInds1[fbrPtr1[fbr]];
      idx0 = fbrLikeSlcInds[fbrS+fr];//slc;
      tmp = 0;

      for (int fbr = fbrPtr1[fbrS+fr] + workId; fbr < fbrPtr1[fbrS+fr+1]; fbr+=warpPerSlice){
        ITYPE idx2 = fbrIdx2[fbr];
        tmp_val = 0;

        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; x++) {

          unsigned int idx3 = dInds3[x];

          for(unsigned int r=laneId; r<R; r+=32) {
            tmp_val += vals[x] * dU3[idx3 * R + r]; //2MR
          }
        }

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp += tmp_val * dU2[idx2 * R + r] ;
        }
      }
      for(unsigned int r=laneId; r<R; r+=32) {
        tmp2 += tmp * dU1[idx1 * R + r] ;
      }

      if(fbrLikeSlcInds[fbrS+fr] != fbrLikeSlcInds[fbrS+fr+1]) {

        diffFiber = true;
        for(unsigned int r=laneId; r<R; r+=32) {
          atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
        }
        tmp2 = 0;
      }
    }

    if(!diffFiber) {
      for(unsigned int r=laneId; r<R; r+=32)
        atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
    }
  }
}

// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
  DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC) {

  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val, tmp2 = 0;

  if(fbrS < nFibers - 1){

    tmp = 0;
    unsigned int idx0 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];
    unsigned int idx3 = fbrLikeSlcInds[fbrS];//slc;

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      unsigned int idx1 = fbrIdx2[fbr];
      tmp_val = 0;

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx2 = dInds3[x];

        for(unsigned int r=laneId; r<R; r+=32)
          tmp_val += vals[x] * dU2[idx2 * R + r] ; //2MR
      }
      for(unsigned int r=laneId; r<R; r+=32)
        tmp += tmp_val * dU1[idx1 * R + r]  ;
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      tmp2 = tmp * dU3[idx3 * R + r];
      atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
    }
  }
}

// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC) {
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){

    tmp_val = 0;
    unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
    unsigned int idx2 = fbrLikeSlcInds[fbr];//slc;

    for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

      unsigned int idx1 = dInds2[x];

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp_val += vals[x] * dU1[idx1 * R + r]; //2MR
      }
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      tmp = tmp_val * dU2[idx2 * R + r] ;
      atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
    }
  }
}

// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC) {
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp;

  if(fbrS < nFibers - 1){

    unsigned int idx2 = fbrLikeSlcInds[fbrS];//slc;
    unsigned int idx3 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      unsigned int idx0 = fbrIdx2[fbr];
      tmp = 0;

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx1 = dInds3[x];

        for(unsigned int r=laneId; r<R; r+=32)
          tmp += vals[x] * dU1[idx1 * R + r]; //2MR
      }
      for(unsigned int r=laneId; r<R; r+=32)  {
        atomicAdd(&dU0[idx0 * R + r], tmp * dU2[idx2 * R + r] * dU3[idx3 * R + r]) ;
      }
    }
  }
}

// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC) {
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){

    tmp_val = 0;
    unsigned int idx1 = fbrLikeSlcInds[fbr];//slc;
    unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];

    for(unsigned int r=laneId; r<R; r+=32)
      tmp = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR

    for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

      unsigned int idx0 = dInds2[x];

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp_val = vals[x] * tmp;///dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //2MR
        atomicAdd(&dU0[idx0 * R + r], tmp_val);
      }
    }
  }
}

// HIP fbr atomic sing slcLikeFbr
__device__ __forceinline__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D_body(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC) {
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val = 0;;

  if(fbrS < nFibers - 1){

    tmp = 0;
    unsigned int idx1 = fbrLikeSlcInds[fbrS];//slc;
    unsigned int idx2 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];

    for(unsigned int r=laneId; r<R; r+=32)
      tmp_val = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      ITYPE idx3 = fbrIdx2[fbr];

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx0 = dInds3[x];

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp = vals[x] * dU3[idx3 * R + r] * tmp_val;//2MR
          atomicAdd(&dU0[idx0 * R + r], tmp);
        }
      }
    }
  }
}

// Explicit offsets avoid installed IRIS HIP shared archs_off slot aliasing.
extern "C" __global__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset7, size_t offset8, size_t offset9) { mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds2+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), nFibers, (DTYPE*)((unsigned char*)dU0+offset7), (const DTYPE*)((unsigned char*)dU1+offset8), (const DTYPE*)((unsigned char*)dU2+offset9), mode, R, warpPerSlice, logOfWPC, fbrPerWarp, logOfFPW); }

extern "C" __global__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset6, size_t offset7, size_t offset9, size_t offset10, size_t offset11, size_t offset12) { mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds3+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), (const ITYPE*)((unsigned char*)fbrPtr2+offset6), (const ITYPE*)((unsigned char*)fbrIdx2+offset7), nFibers, (DTYPE*)((unsigned char*)dU0+offset9), (const DTYPE*)((unsigned char*)dU1+offset10), (const DTYPE*)((unsigned char*)dU2+offset11), (const DTYPE*)((unsigned char*)dU3+offset12), mode, R, warpPerSlice, logOfWPC, fbrPerWarp, logOfFPW); }

extern "C" __global__ void mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
  DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset6, size_t offset7, size_t offset9, size_t offset10, size_t offset11, size_t offset12) { mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds3+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), (const ITYPE*)((unsigned char*)fbrPtr2+offset6), (const ITYPE*)((unsigned char*)fbrIdx2+offset7), nFibers, (DTYPE*)((unsigned char*)dU0+offset9), (const DTYPE*)((unsigned char*)dU1+offset10), (const DTYPE*)((unsigned char*)dU2+offset11), (const DTYPE*)((unsigned char*)dU3+offset12), mode, R, warpPerSlice, logOfWPC); }

extern "C" __global__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset7, size_t offset8, size_t offset9) { mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds2+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), nFibers, (DTYPE*)((unsigned char*)dU0+offset7), (const DTYPE*)((unsigned char*)dU1+offset8), (const DTYPE*)((unsigned char*)dU2+offset9), mode, R, warpPerSlice, logOfWPC); }

extern "C" __global__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset6, size_t offset7, size_t offset9, size_t offset10, size_t offset11, size_t offset12) { mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds3+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), (const ITYPE*)((unsigned char*)fbrPtr2+offset6), (const ITYPE*)((unsigned char*)fbrIdx2+offset7), nFibers, (DTYPE*)((unsigned char*)dU0+offset9), (const DTYPE*)((unsigned char*)dU1+offset10), (const DTYPE*)((unsigned char*)dU2+offset11), (const DTYPE*)((unsigned char*)dU3+offset12), mode, R, warpPerSlice, logOfWPC); }

extern "C" __global__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset7, size_t offset8, size_t offset9) { mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds2+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), nFibers, (DTYPE*)((unsigned char*)dU0+offset7), (const DTYPE*)((unsigned char*)dU1+offset8), (const DTYPE*)((unsigned char*)dU2+offset9), mode, R, warpPerSlice, logOfWPC); }

extern "C" __global__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3,
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC, size_t offset0, size_t offset1, size_t offset2, size_t offset3, size_t offset4, size_t offset5, size_t offset6, size_t offset7, size_t offset9, size_t offset10, size_t offset11, size_t offset12) { mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D_body((const DTYPE*)((unsigned char*)vals+offset0), (const ITYPE*)((unsigned char*)fbrLikeSlcInds+offset1), (const ITYPE*)((unsigned char*)dInds3+offset2), (const ITYPE*)((unsigned char*)fbrPtr0+offset3), (const ITYPE*)((unsigned char*)fbrPtr1+offset4), (const ITYPE*)((unsigned char*)fbrIdx1+offset5), (const ITYPE*)((unsigned char*)fbrPtr2+offset6), (const ITYPE*)((unsigned char*)fbrIdx2+offset7), nFibers, (DTYPE*)((unsigned char*)dU0+offset9), (const DTYPE*)((unsigned char*)dU1+offset10), (const DTYPE*)((unsigned char*)dU2+offset11), (const DTYPE*)((unsigned char*)dU3+offset12), mode, R, warpPerSlice, logOfWPC); }
