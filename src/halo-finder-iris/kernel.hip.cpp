#include <hip/hip_runtime.h>
#include "BasicDefinition.h"
#define TILEX 4
#define TILEY 4
#define BLOCKX 32
#define BLOCKY 4
#define MAXX 32
#define MAXY 256
__device__  __forceinline__
void atomicWarpReduceAndUpdate(POSVEL_T *out, POSVEL_T val) {
  //perform shfl reduction
  val+=__shfl_down(val, 16, 32);
  val+=__shfl_down(val, 8, 32);
  val+=__shfl_down(val, 4, 32);
  val+=__shfl_down(val, 2, 32);
  val+=__shfl_down(val, 1, 32);

  if(hipThreadIdx_x%32==0)
    atomicAdd(out,val);  //atomics are unecessary but they are faster than non-atomics due to a single bus transaction
}


typedef long long int int64;

template<typename T>
__device__ __forceinline__
T load(T *t)
{
  return __ldg(t);  //texture load
}

template<int TILE_SIZE, typename T>__device__ void loadT(T *  out, const T *  in);

//generic version (inefficient)
template<int TILE_SIZE, typename T>
__device__ __forceinline__
void loadT(T *  out, const T *  in) {
  #pragma unroll
  for(int i=0;i<TILE_SIZE;i++) {
    out[i]=__ldg(in+i);
  }
}

//Vector loads
template<>
__device__ __forceinline__
void loadT<2,float>(float *  out, const float *  in) {
  *reinterpret_cast<float2*>(out)=load(reinterpret_cast<const float2*>(in));
}
template<>
__device__ __forceinline__
void loadT<4,float>(float *  out, const float *  in) {
  *reinterpret_cast<float4*>(out)=load(reinterpret_cast<const float4*>(in));
}

//static __device__ __forceinline__ float __internal_fast_rsqrtf(float a)
//{
//  float r;
//  asm ("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
//  return r;
//}

//computes the forces between tiles i and j, adds the change in force to xi,yi,zi
template<int TX, int TY>
__device__ __forceinline__ void computeForces(POSVEL_T xxi[], POSVEL_T yyi[], POSVEL_T zzi[],
                                              POSVEL_T xxj[], POSVEL_T yyj[], POSVEL_T zzj[], POSVEL_T massj[],
                                              POSVEL_T xi[], POSVEL_T yi[], POSVEL_T zi[],
                                              POSVEL_T ma0, POSVEL_T ma1, POSVEL_T ma2, POSVEL_T ma3, POSVEL_T ma4, POSVEL_T ma5,
                                              POSVEL_T mp_rsm2, POSVEL_T fsrrmax2) {

  #pragma unroll
  for(int i=0;i<TY;i++) {
    #pragma unroll
    for(int j=0;j<TX;j++) {
      POSVEL_T dxc = xxj[j] - xxi[i];                                                                //1 FADD
      POSVEL_T dyc = yyj[j] - yyi[i];                                                                //1 FADD
      POSVEL_T dzc = zzj[j] - zzi[i];                                                                //1 FADD

      POSVEL_T r2 = dxc * dxc + dyc * dyc + dzc * dzc;                                               //1 FMUL 2 FMA
      POSVEL_T v=r2+mp_rsm2;                                                                         //1 FADD
      POSVEL_T v3=v*v*v;                                                                             //2 FMUL

      POSVEL_T f = __frsqrt_rn(v3);                                                        //1 MUFU,
	  // MDS: Should ask someone why this line is dangling
      //       - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5)))));                          //5 FMA, 1 FADD
//#define BUG
#ifndef BUG
      f*=massj[j]*(r2<fsrrmax2 && r2>0.0f);                                                          //2 FMUL, 1 FSETP, 1 FCMP
#else
      f*=massj[j];                                                                                   //1 FMUL
      f*=(r2<fsrrmax2 && r2>0.0f);                                                                   //1 FMUL, 1 FSETP, 1 FCMP
#endif

      xi[i] = xi[i] + f * dxc;                                                                       //1 FMA
      yi[i] = yi[i] + f * dyc;                                                                       //1 FMA
      zi[i] = zi[i] + f * dzc;                                                                       //1 FMA
    }
  }
}

//loads a tile from memory.  Use checkBounds and loadMass to disable bounds check or mass load at compile time
template <bool checkBounds, bool loadMass, int TILE_SIZE>
__device__ __forceinline__
void loadTile(int i, int bounds,
              const POSVEL_T*  xx, const POSVEL_T*  yy, const POSVEL_T*  zz, const POSVEL_T*  mass,
              POSVEL_T xxi[], POSVEL_T yyi[], POSVEL_T zzi[], POSVEL_T massi[]) {
  if(checkBounds) {
  #pragma unroll
  for(int64 u=0;u<TILE_SIZE;u++) {
    int64 idx=TILE_SIZE*i+u;                                                                        // 1 IMAD

#if 1
      bool cond=idx<bounds;
      xxi[u] = (cond) ? load(xx+idx) : 0.0f;                                                     // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
      yyi[u] = (cond) ? load(yy+idx) : 0.0f;                                                     // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
      zzi[u] = (cond) ? load(zz+idx) : 0.0f;                                                     // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
      if(loadMass) massi[u] = (cond) ? load(mass+idx) : 0.0f;                                    // 1 ISETP, 1 LDG, 2 IMAD, 1 MOV
#else
      massi[u] = 0.0f;                                                                           //1 MOV
      if(idx<bounds) {                                                                           //1 ISETP, 1 BRA
        xxi[u] = load(xx+idx);                                                                   //1 LDG, 2 IMAD
        yyi[u] = load(yy+idx);                                                                   //1 LDG, 2 IMAD
        zzi[u] = load(zz+idx);                                                                   //1 LDG, 2 IMAD
        if(loadMass) massi[u] = load(mass+idx);                                                  //1 LDG, 2 IMAD
      }
#endif
    }
  } else {

    int idx=TILE_SIZE*i;
    loadT<TILE_SIZE>(xxi,xx+idx);                                                                //1 LDG, 2 IMAD
    loadT<TILE_SIZE>(yyi,yy+idx);                                                                //1 LDG, 2 IMAD
    loadT<TILE_SIZE>(zzi,zz+idx);                                                                //1 LDG, 2 IMAD
    if(loadMass) loadT<TILE_SIZE>(massi,mass+idx);                                               //1 LDG, 2 IMAD
  }
}

//applies the force in xi,yi,zi to update vx, vy, vz
//use checkBounds to disable bounds checking at compile time
template <bool checkBounds, int TILE_SIZE>
__device__ __forceinline__
void applyForce(int i, int bounds,POSVEL_T fcoeff,
                const POSVEL_T xi[], const POSVEL_T yi[], const POSVEL_T zi[],
                POSVEL_T *vx, POSVEL_T *vy, POSVEL_T *vz) {
    #pragma unroll
    for(int u=0;u<TILE_SIZE;u++) {
      int idx=TILE_SIZE*i+u;                                                                         //1 IMAD

      if(!checkBounds || idx<bounds)
      {                                                                                           //1 ISETP
        atomicWarpReduceAndUpdate(vx+idx,fcoeff * xi[u]);                                         //2 IMAD, 6 FADD
        atomicWarpReduceAndUpdate(vy+idx,fcoeff * yi[u]);                                         //2 IMAD, 6 FADD
        atomicWarpReduceAndUpdate(vz+idx,fcoeff * zi[u]);                                         //2 IMAD, 6 FADD
      }
    }
}

  //Tell the compiler how many blocks we expect to be active.
  //This gives the compiler a better idea of how many registers to use.  The second number is tunable.
__launch_bounds__(BLOCKX*BLOCKY,7)
extern "C" __global__
void Step10_kernel(int count, int count1,
                   const POSVEL_T*  xx, const POSVEL_T*  yy,
                   const POSVEL_T*  zz, const POSVEL_T*  mass,
                   const POSVEL_T*  xx1, const POSVEL_T*  yy1,
                   const POSVEL_T*  zz1, const POSVEL_T*  mass1,
                   POSVEL_T*  vx, POSVEL_T*  vy,
                   POSVEL_T*  vz, POSVEL_T fsrrmax2, POSVEL_T mp_rsm2, POSVEL_T fcoeff)
{
  const POSVEL_T ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;

  //Register arrays to hold tiles of data
  POSVEL_T xxi[TILEY];
  POSVEL_T yyi[TILEY];
  POSVEL_T zzi[TILEY];
  POSVEL_T xxj[TILEX];
  POSVEL_T yyj[TILEX];
  POSVEL_T zzj[TILEX];
  POSVEL_T massj[TILEX];

  // Consolidate variables to help fit within the register limit
  int x_idx = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
  int y_idx = hipBlockIdx_y*hipBlockDim_y+hipThreadIdx_y;

  //loop over interior region and calculate forces.
  //for each tile i
 for(int i=y_idx;i<count/TILEY;i+=hipBlockDim_y*hipGridDim_y)                                //1 ISETP
  {
    POSVEL_T xi[TILEY]={0};                                                                                //TILEY MOV
    POSVEL_T yi[TILEY]={0};                                                                                //TILEY MOV
    POSVEL_T zi[TILEY]={0};                                                                                //TILEY MOV

    //load tile i,mass and bounds check are not needed
    loadTile<false,false,TILEY>(i,count,xx,yy,zz,NULL,xxi,yyi,zzi,NULL);

    //for each tile j
    for (int j=x_idx;j<count1/TILEX;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is not needed
      loadTile<false,true,TILEX>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<TILEX,TILEY>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    //process remaining elements at the end, use TILEX=1
    for (int j=count1/TILEX*TILEX+x_idx;j<count1;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is needed, mass is needed
      loadTile<true,true,1>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<1,TILEY>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    //apply the force we have calculated above, bounds check is not needed
    applyForce<false,TILEY>(i,count,fcoeff,xi,yi,zi,vx,vy,vz);
  }

  //At this point we have computed almost all interactions.
  //However we still need to add contributions for particles at the end

#if 1
  //process ramining elements in set TILEY=1
  //for each tile i
  for(int i=y_idx;i<count - count/TILEY*TILEY;i+=hipBlockDim_y*hipGridDim_y)                             //1 ISETP
  {
    // Taken out of the loop condition to help fit within the register limit
    int k = i + count/TILEY*TILEY;
    POSVEL_T xi[1]={0};                                                                                //1 MOV
    POSVEL_T yi[1]={0};                                                                                //1 MOV
    POSVEL_T zi[1]={0};                                                                                //1 MOV

    //load xxi, yyi, zzi tiles, mass is not needed, bounds check is needed
    loadTile<true,false,1>(k,count,xx,yy,zz,NULL,xxi,yyi,zzi,NULL);

    //for each tile j
    for (int j=x_idx;j<count1/TILEX;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is not needed
      loadTile<false,true,TILEX>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<TILEX,1>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    //process remaining elements at the end, use TILEX=1
    for (int j=count1/TILEX*TILEX+x_idx;j<count1;j+=hipBlockDim_x*hipGridDim_x)                                  //1 ISETP
    {
      //load tile j, bounds check is needed, mass is needed
      loadTile<true,true,1>(j,count1,xx1,yy1,zz1,mass1,xxj,yyj,zzj,massj);

      //compute forces between tile i and tile j
      computeForces<1,1>(xxi,yyi,zzi,xxj,yyj,zzj,massj,xi,yi,zi,ma0,ma1,ma2,ma3,ma4,ma5,mp_rsm2,fsrrmax2);
    }

    applyForce<true,1>(k,count,fcoeff,xi,yi,zi,vx,vy,vz);
  }
#endif

}
