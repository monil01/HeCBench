#include <hip/hip_runtime.h>
inline __device__ float4 sqrtf(float4 v)
{
    return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}

inline __device__ float4 fast_sqrtf(float4 v)
{
    return make_float4(__fsqrt_rn(v.x), __fsqrt_rn(v.y), __fsqrt_rn(v.z), __fsqrt_rn(v.w));
}

inline __device__ float4 expf(float4 v)
{
    return make_float4(expf(v.x), expf(v.y), expf(v.z), expf(v.w));
}

inline __device__ float4 fast_expf(float4 v)
{
    return make_float4(__expf(v.x), __expf(v.y), __expf(v.z), __expf(v.w));
}

extern "C" __global__ void mdh (
    const float *__restrict__ ax, 
    const float *__restrict__ ay,
    const float *__restrict__ az,
    const float *__restrict__ gx, 
    const float *__restrict__ gy, 
    const float *__restrict__ gz,
    const float *__restrict__ charge, 
    const float *__restrict__ size, 
          float *__restrict__ val,
    const float pre1,
    const float xkappa, 
    const int natom)
{
  __shared__ float shared[5*1024];

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid * 5    ] = ax[jatom + lid];
      shared[lid * 5 + 1] = ay[jatom + lid];
      shared[lid * 5 + 2] = az[jatom + lid];
      shared[lid * 5 + 3] = charge[jatom + lid];
      shared[lid * 5 + 4] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i * 5    ];
      float4 dy = lgy - shared[i * 5 + 1];
      float4 dz = lgz - shared[i * 5 + 2];
      float4 dist = sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * (shared[i * 5 + 3] / dist)  *
           expf( -xkappa * (dist - shared[i * 5 + 4])) /
           (1.0f + xkappa * shared[i * 5 + 4]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

extern "C" __global__ void mdh2 (
    const float *__restrict__ ax, 
    const float *__restrict__ ay,
    const float *__restrict__ az,
    const float *__restrict__ gx, 
    const float *__restrict__ gy, 
    const float *__restrict__ gz,
    const float *__restrict__ charge, 
    const float *__restrict__ size, 
          float *__restrict__ val,
    const float pre1, 
    const float xkappa, 
    const int natom)
{
  __shared__ float shared[5*1024];

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid          ] = ax[jatom + lid];
      shared[lid +   lsize] = ay[jatom + lid];
      shared[lid + 2*lsize] = az[jatom + lid];
      shared[lid + 3*lsize] = charge[jatom + lid];
      shared[lid + 4*lsize] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i          ];
      float4 dy = lgy - shared[i +   lsize];
      float4 dz = lgz - shared[i + 2*lsize];
      float4 dist = sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * ( shared[i + 3*lsize] / dist )  *
           expf( -xkappa * (dist - shared[i + 4*lsize])) /
           (1.0f + xkappa * shared[i + 4*lsize]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

extern "C" __global__ void mdh3 (
    const float *__restrict__ ax, 
    const float *__restrict__ ay,
    const float *__restrict__ az,
    const float *__restrict__ gx, 
    const float *__restrict__ gy, 
    const float *__restrict__ gz,
    const float *__restrict__ charge, 
    const float *__restrict__ size, 
          float *__restrict__ val,
    const float pre1, 
    const float xkappa, 
    const int natom)
{
  __shared__ float shared[5*1024];

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid          ] = ax[jatom + lid];
      shared[lid +   lsize] = ay[jatom + lid];
      shared[lid + 2*lsize] = az[jatom + lid];
      shared[lid + 3*lsize] = charge[jatom + lid];
      shared[lid + 4*lsize] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i          ];
      float4 dy = lgy - shared[i +   lsize];
      float4 dz = lgz - shared[i + 2*lsize];
      float4 dist = fast_sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * ( shared[i + 3*lsize] / dist )  *
           fast_expf( -xkappa * (dist - shared[i + 4*lsize])) /
           (1.0f + xkappa * shared[i + 4*lsize]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}
