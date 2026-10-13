#include <hip/hip_runtime.h>
#include "device_params.h"
#define LOCAL_SIZE_LIMIT 512U
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
inline __device__ void operator+=(float4 &a, float4 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

inline __device__ float4 operator+(float4 a, float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __device__ int4 operator+(int4 a, int4 b)
{
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __device__ float4 operator*(float4 a, float4 b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

inline __device__ float4 operator-(float4 a, float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __device__ float4 operator*(float4 a, float b)
{
  return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline __device__ float4 operator*(float b, float4 a)
{
  return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline __device__ void operator*=(float4 &a, const float b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// Euler integration
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void integrateSystemK(
    float4*__restrict__ d_Pos,  //input/output
    float4*__restrict__ d_Vel,  //input/output
    const simParams_t params,
    const float deltaTime,
    const unsigned int numParticles)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= numParticles) return;

  float4 pos = d_Pos[index];
  float4 vel = d_Vel[index];

  pos.w = 1.0f;
  vel.w = 0.0f;

  //Gravity
  float4 g = {params.gravity.x, params.gravity.y, params.gravity.z, 0};
  vel += g * deltaTime;
  vel *= params.globalDamping;

  //Advance pos
  pos += vel * deltaTime;

  //printf("before %d %3.f %3.f %3.f\n", index, pos.x, pos.y, pos.z);

  //Collide with cube
  if(pos.x < -1.0f + params.particleRadius){
    pos.x = -1.0f + params.particleRadius;
    vel.x *= params.boundaryDamping;
  }
  if(pos.x > 1.0f - params.particleRadius){
    pos.x = 1.0f - params.particleRadius;
    vel.x *= params.boundaryDamping;
  }

  if(pos.y < -1.0f + params.particleRadius){
    pos.y = -1.0f + params.particleRadius;
    vel.y *= params.boundaryDamping;
  }
  if(pos.y > 1.0f - params.particleRadius){
    pos.y = 1.0f - params.particleRadius;
    vel.y *= params.boundaryDamping;
  }

  if(pos.z < -1.0f + params.particleRadius){
    pos.z = -1.0f + params.particleRadius;
    vel.z *= params.boundaryDamping;
  }
  if(pos.z > 1.0f - params.particleRadius){
    pos.z = 1.0f - params.particleRadius;
    vel.z *= params.boundaryDamping;
  }

  //Store new position and velocity
  d_Pos[index] = pos;
  d_Vel[index] = vel;
  //printf("after %d %3.f %3.f %3.f\n", index, pos.x, pos.y, pos.z);
}

////////////////////////////////////////////////////////////////////////////////
// Save particle grid cell hashes and indices
////////////////////////////////////////////////////////////////////////////////
__device__
int4 getGridPos(const float4 p, const simParams_t &params)
{
  int4 gridPos;
  gridPos.x = (int)floor((p.x - params.worldOrigin.x) / params.cellSize.x);
  gridPos.y = (int)floor((p.y - params.worldOrigin.y) / params.cellSize.y);
  gridPos.z = (int)floor((p.z - params.worldOrigin.z) / params.cellSize.z);
  gridPos.w = 0;
  return gridPos;
}

//Calculate address in grid from position (clamping to edges)
__device__
unsigned int getGridHash(int4 gridPos, const simParams_t &params)
{
  //Wrap addressing, assume power-of-two grid dimensions
  gridPos.x = gridPos.x & (params.gridSize.x - 1);
  gridPos.y = gridPos.y & (params.gridSize.y - 1);
  gridPos.z = gridPos.z & (params.gridSize.z - 1);
  return UMAD( UMAD(gridPos.z, params.gridSize.y, gridPos.y), params.gridSize.x, gridPos.x );
}

//Calculate grid hash value for each particle
extern "C" __global__ void calcHashK(
    unsigned int*__restrict__ d_Hash, //output
    unsigned int*__restrict__ d_Index, //output
    const float4*__restrict__ d_Pos, //input: positions
    const simParams_t params,
    unsigned int numParticles)
{
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= numParticles) return;

  float4 p = d_Pos[index];

  //Get address in grid
  int4  gridPos = getGridPos(p, params);
  unsigned int gridHash = getGridHash(gridPos, params);

  //Store grid hash and particle index
  d_Hash[index] = gridHash;
  d_Index[index] = index;
}

////////////////////////////////////////////////////////////////////////////////
// Find cell bounds and reorder positions+velocities by sorted indices
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void memSetK(
    unsigned int* d_Data,
    const unsigned int val,
    const unsigned int N)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) d_Data[i] = val;
}

extern "C" __global__ void findCellBoundsAndReorderK(
    unsigned int*__restrict__ d_CellStart,     //output: cell start index
    unsigned int*__restrict__ d_CellEnd,       //output: cell end index
    float4*__restrict__ d_ReorderedPos,  //output: reordered by cell hash positions
    float4*__restrict__ d_ReorderedVel,  //output: reordered by cell hash velocities
    const unsigned int*__restrict__ d_Hash,    //input: sorted grid hashes
    const unsigned int*__restrict__ d_Index,   //input: particle indices sorted by hash
    const float4*__restrict__ d_Pos,     //input: positions array sorted by hash
    const float4*__restrict__ d_Vel,     //input: velocity array sorted by hash
    const unsigned int numParticles)
{
  unsigned int hash;
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int lid = threadIdx.x;

  __shared__ unsigned int localHash[65];

  //Handle case when no. of particles not multiple of block size
  if(index < numParticles){
    hash = d_Hash[index];

    //Load hash data into local memory so that we can look 
    //at neighboring particle's hash value without loading
    //two hash values per thread
    localHash[lid + 1] = hash;

    //First thread in block must load neighbor particle hash
    if(index > 0 && lid == 0)
      localHash[0] = d_Hash[index - 1];
  }

  __syncthreads();

  if(index < numParticles){
    //Border case
    if(index == 0)
      d_CellStart[hash] = 0;

    //Main case
    else{
      if(hash != localHash[lid])
        d_CellEnd[localHash[lid]]  = d_CellStart[hash] = index;
    };

    //Another border case
    if(index == numParticles - 1)
      d_CellEnd[hash] = numParticles;


    //Now use the sorted index to reorder the pos and vel arrays
    unsigned int sortedIndex = d_Index[index];
    float4 pos = d_Pos[sortedIndex];
    float4 vel = d_Vel[sortedIndex];

    d_ReorderedPos[index] = pos;
    d_ReorderedVel[index] = vel;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Process collisions (calculate accelerations)
////////////////////////////////////////////////////////////////////////////////
__device__
float4 collideSpheres(
    float4 posA,
    float4 posB,
    float4 velA,
    float4 velB,
    float radiusA,
    float radiusB,
    float spring,
    float damping,
    float shear,
    float attraction)
{
  //Calculate relative position
  float4     relPos = {posB.x - posA.x, posB.y - posA.y, posB.z - posA.z, 0};
  float        dist = sqrt(relPos.x * relPos.x + relPos.y * relPos.y + relPos.z * relPos.z);
  float collideDist = radiusA + radiusB;

  float4 force = {0, 0, 0, 0};
  if(dist < collideDist){
    float4 norm = {relPos.x / dist, relPos.y / dist, relPos.z / dist, 0};

    //Relative velocity
    float4 relVel = {velB.x - velA.x, velB.y - velA.y, velB.z - velA.z, 0};

    //Relative tangential velocity
    float relVelDotNorm = relVel.x * norm.x + relVel.y * norm.y + relVel.z * norm.z;
    float4 tanVel = {relVel.x - relVelDotNorm * norm.x, relVel.y - relVelDotNorm * norm.y, 
      relVel.z - relVelDotNorm * norm.z, 0};

    //Spring force (potential)
    float springFactor = -spring * (collideDist - dist);
    force = {
      springFactor * norm.x + damping * relVel.x + shear * tanVel.x + attraction * relPos.x,
      springFactor * norm.y + damping * relVel.y + shear * tanVel.y + attraction * relPos.y,
      springFactor * norm.z + damping * relVel.z + shear * tanVel.z + attraction * relPos.z,
      0
    };
  }

  return force;
}

extern "C" __global__ void collideK(
    float4*__restrict__ d_Vel,          //output: new velocity
    const float4*__restrict__ d_ReorderedPos, //input: reordered positions
    const float4*__restrict__ d_ReorderedVel, //input: reordered velocities
    const unsigned int*__restrict__ d_Index,        //input: reordered particle indices
    const unsigned int*__restrict__ d_CellStart,    //input: cell boundaries
    const unsigned int*__restrict__ d_CellEnd,
    const simParams_t params,
    const unsigned int numParticles)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= numParticles) return;

  float4   pos = d_ReorderedPos[index];
  float4   vel = d_ReorderedVel[index];
  float4 force = {0, 0, 0, 0};

  //Get address in grid
  int4 gridPos = getGridPos(pos, params);

  //Accumulate surrounding cells
  for(int z = -1; z <= 1; z++)
    for(int y = -1; y <= 1; y++)
      for(int x = -1; x <= 1; x++){
        //Get start particle index for this cell
        int4 t = {x, y, z, 0};
        unsigned int   hash = getGridHash(gridPos + t, params);
        unsigned int startI = d_CellStart[hash];

        //Skip empty cell
        if(startI == 0xFFFFFFFFU) continue;

        //Iterate over particles in this cell
        unsigned int endI = d_CellEnd[hash];
        for(unsigned int j = startI; j < endI; j++){
          if(j == index) continue;

          float4 pos2 = d_ReorderedPos[j];
          float4 vel2 = d_ReorderedVel[j];

          //Collide two spheres
          force += collideSpheres(
              pos, pos2,
              vel, vel2,
              params.particleRadius, params.particleRadius, 
              params.spring, params.damping, params.shear, params.attraction);
        }
      }

  //Collide with cursor sphere
  force += collideSpheres(
      pos, {params.colliderPos.x, params.colliderPos.y, params.colliderPos.z, 0},
      vel, {0, 0, 0, 0},
      params.particleRadius, params.colliderRadius,
      params.spring, params.damping, params.shear, params.attraction);

  //Write new velocity back to original unsorted location
  d_Vel[d_Index[index]] = vel + force;
}
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

__device__
inline void ComparatorPrivate(
    unsigned int *keyA,
    unsigned int *valA,
    unsigned int *keyB,
    unsigned int *valB,
    unsigned int dir)
{
  if( (*keyA > *keyB) == dir ){
    unsigned int t;
    t = *keyA; *keyA = *keyB; *keyB = t;
    t = *valA; *valA = *valB; *valB = t;
  }
}

__device__
inline void ComparatorLocal(
    unsigned int* keyA,
    unsigned int* valA,
    unsigned int* keyB,
    unsigned int* valB,
    const unsigned int dir)
{
  if( (*keyA > *keyB) == dir ){
    unsigned int t;
    t = *keyA; *keyA = *keyB; *keyB = t;
    t = *valA; *valA = *valB; *valB = t;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into local memory
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void bitonicSortLocal(
    unsigned int*__restrict__ d_DstKey,
    unsigned int*__restrict__ d_DstVal,
    const unsigned int*__restrict__ d_SrcKey,
    const unsigned int*__restrict__ d_SrcVal,
    const unsigned int arrayLength,
    const unsigned int dir)
{
  __shared__  unsigned int l_key[LOCAL_SIZE_LIMIT];
  __shared__  unsigned int l_val[LOCAL_SIZE_LIMIT];

  //Offset to the beginning of subbatch and load data
  d_SrcKey += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  l_key[threadIdx.x +                      0] = d_SrcKey[                     0];
  l_val[threadIdx.x +                      0] = d_SrcVal[                     0];
  l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
  l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

  for(unsigned int size = 2; size < arrayLength; size <<= 1){
    //Bitonic merge
    unsigned int ddd = dir ^ ( (threadIdx.x & (size / 2)) != 0 );
    for(unsigned int stride = size / 2; stride > 0; stride >>= 1){
      __syncthreads();
      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      ComparatorLocal(
          &l_key[pos +      0], &l_val[pos +      0],
          &l_key[pos + stride], &l_val[pos + stride],
          ddd);
    }
  }

  //ddd == dir for the last bitonic merge step
  {
    for(unsigned int stride = arrayLength / 2; stride > 0; stride >>= 1){
      __syncthreads();
      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      ComparatorLocal(
          &l_key[pos +      0], &l_val[pos +      0],
          &l_key[pos + stride], &l_val[pos + stride],
          dir);
    }
  }

  __syncthreads();
  d_DstKey[                     0] = l_key[threadIdx.x +                      0];
  d_DstVal[                     0] = l_val[threadIdx.x +                      0];
  d_DstKey[(LOCAL_SIZE_LIMIT / 2)] = l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
  d_DstVal[(LOCAL_SIZE_LIMIT / 2)] = l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into local memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortLocal with the only exception
//of even / odd subarrays (of LOCAL_SIZE_LIMIT points) being
//sorted in opposite directions
extern "C" __global__ void bitonicSortLocal1(
    unsigned int*__restrict__ d_DstKey,
    unsigned int*__restrict__ d_DstVal,
    const unsigned int*__restrict__ d_SrcKey,
    const unsigned int*__restrict__ d_SrcVal)
{
  __shared__ unsigned int l_key[LOCAL_SIZE_LIMIT];
  __shared__ unsigned int l_val[LOCAL_SIZE_LIMIT];

  //Offset to the beginning of subarray and load data
  d_SrcKey += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  l_key[threadIdx.x +                      0] = d_SrcKey[                     0];
  l_val[threadIdx.x +                      0] = d_SrcVal[                     0];
  l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
  l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

  unsigned int comparatorI = (blockIdx.x * blockDim.x + threadIdx.x) & ((LOCAL_SIZE_LIMIT / 2) - 1);

  for(unsigned int size = 2; size < LOCAL_SIZE_LIMIT; size <<= 1){
    //Bitonic merge
    unsigned int ddd = (comparatorI & (size / 2)) != 0;
    for(unsigned int stride = size / 2; stride > 0; stride >>= 1){
      __syncthreads();
      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      ComparatorLocal(
          &l_key[pos +      0], &l_val[pos +      0],
          &l_key[pos + stride], &l_val[pos + stride],
          ddd
               );
    }
  }

  //Odd / even arrays of LOCAL_SIZE_LIMIT elements
  //sorted in opposite directions
  {
    unsigned int ddd = (blockIdx.x & 1);
    for(unsigned int stride = LOCAL_SIZE_LIMIT / 2; stride > 0; stride >>= 1){
      __syncthreads();
      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      ComparatorLocal(
          &l_key[pos +      0], &l_val[pos +      0],
          &l_key[pos + stride], &l_val[pos + stride],
          ddd
               );
    }
  }

  __syncthreads();
  d_DstKey[                     0] = l_key[threadIdx.x +                      0];
  d_DstVal[                     0] = l_val[threadIdx.x +                      0];
  d_DstKey[(LOCAL_SIZE_LIMIT / 2)] = l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
  d_DstVal[(LOCAL_SIZE_LIMIT / 2)] = l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for 'stride' >= LOCAL_SIZE_LIMIT
extern "C" __global__ void bitonicMergeGlobal(
    unsigned int*__restrict__ d_DstKey,
    unsigned int*__restrict__ d_DstVal,
    const unsigned int*__restrict__ d_SrcKey,
    const unsigned int*__restrict__ d_SrcVal,
    const unsigned int arrayLength,
    const unsigned int size,
    const unsigned int stride,
    const unsigned int dir)
{
  unsigned int global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

  //Bitonic merge
  unsigned int ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
  unsigned int pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

  unsigned int keyA = d_SrcKey[pos +      0];
  unsigned int valA = d_SrcVal[pos +      0];
  unsigned int keyB = d_SrcKey[pos + stride];
  unsigned int valB = d_SrcVal[pos + stride];

  ComparatorPrivate(
      &keyA, &valA,
      &keyB, &valB,
      ddd);

  d_DstKey[pos +      0] = keyA;
  d_DstVal[pos +      0] = valA;
  d_DstKey[pos + stride] = keyB;
  d_DstVal[pos + stride] = valB;
}

//Combined bitonic merge steps for
//'size' > LOCAL_SIZE_LIMIT and 'stride' = [1 .. LOCAL_SIZE_LIMIT / 2]
extern "C" __global__ void bitonicMergeLocal(
    unsigned int*__restrict__ d_DstKey,
    unsigned int*__restrict__ d_DstVal,
    const unsigned int*__restrict__ d_SrcKey,
    const unsigned int*__restrict__ d_SrcVal,
    const unsigned int arrayLength,
    const unsigned int size,
    unsigned int stride,
    const unsigned int dir)
{
  __shared__ unsigned int l_key[LOCAL_SIZE_LIMIT];
  __shared__ unsigned int l_val[LOCAL_SIZE_LIMIT];

  d_SrcKey += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * LOCAL_SIZE_LIMIT + threadIdx.x;
  l_key[threadIdx.x +                      0] = d_SrcKey[                     0];
  l_val[threadIdx.x +                      0] = d_SrcVal[                     0];
  l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] = d_SrcKey[(LOCAL_SIZE_LIMIT / 2)];
  l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)] = d_SrcVal[(LOCAL_SIZE_LIMIT / 2)];

  //Bitonic merge
  unsigned int comparatorI = (blockIdx.x * blockDim.x + threadIdx.x) & ((arrayLength / 2) - 1);
  unsigned int         ddd = dir ^ ( (comparatorI & (size / 2)) != 0 );
  for(; stride > 0; stride >>= 1){
    __syncthreads();
    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    ComparatorLocal(
        &l_key[pos +      0], &l_val[pos +      0],
        &l_key[pos + stride], &l_val[pos + stride],
        ddd);
  }

  __syncthreads();
  d_DstKey[                     0] = l_key[threadIdx.x +                      0];
  d_DstVal[                     0] = l_val[threadIdx.x +                      0];
  d_DstKey[(LOCAL_SIZE_LIMIT / 2)] = l_key[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
  d_DstVal[(LOCAL_SIZE_LIMIT / 2)] = l_val[threadIdx.x + (LOCAL_SIZE_LIMIT / 2)];
}

