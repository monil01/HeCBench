#ifndef PARTICLES_H
#define PARTICLES_H
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
struct float3 {float x,y,z;};
struct uint3 {unsigned int x,y,z;};
struct alignas(16) float4 {float x,y,z,w;};
template<class T> int info(T&) {return sizeof(T);}
inline int info(iris::DMem&) {return iris_rw;}
template<class... A> void launch(const char* name,size_t global,size_t local,A&... a) {
 void* params[]={const_cast<void*>(static_cast<const void*>(&a))...};
 int flags[]={info(a)...}; size_t gws[]={global},lws[]={local}; iris::Task t;
 if(t.kernel(name,1,nullptr,gws,lws,sizeof...(A),params,flags)!=IRIS_SUCCESS ||
 t.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}

#define UMAD(a, b, c)  ( (a) * (b) + (c) )

//Simulation parameters
typedef struct{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    unsigned int numCells;
    float3 worldOrigin;
    float3 cellSize;

    unsigned int numBodies;
    unsigned int maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} simParams_t;

void integrateSystem(
    iris::DMem& d_Pos,
    iris::DMem& d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles);

void calcHash(
    iris::DMem& d_Hash,
    iris::DMem& d_Index,
    iris::DMem& d_Pos,
    const simParams_t &params,
    const int numParticles);

void memSet(
    iris::DMem& d_Data,
    unsigned int val,
    unsigned int N);

void findCellBoundsAndReorder(
    iris::DMem& d_CellStart,
    iris::DMem& d_CellEnd,
    iris::DMem& d_ReorderedPos,
    iris::DMem& d_ReorderedVel,
    iris::DMem& d_Hash,
    iris::DMem& d_Index,
    iris::DMem& d_Pos,
    iris::DMem& d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells);

void collide(
    iris::DMem& d_Vel,
    iris::DMem& d_ReorderedPos,
    iris::DMem& d_ReorderedVel,
    iris::DMem& d_Index,
    iris::DMem& d_CellStart,
    iris::DMem& d_CellEnd,
    const simParams_t &params,
    const unsigned int   numParticles,
    const unsigned int   numCells);

void bitonicSort(
    iris::DMem& d_DstKey,
    iris::DMem& d_DstVal,
    iris::DMem& d_SrcKey,
    iris::DMem& d_SrcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir);
#endif

