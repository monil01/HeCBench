#ifndef PARTICLES_H
#define PARTICLES_H

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

static_assert(sizeof(simParams_t)==104,"parameter ABI");
#endif

