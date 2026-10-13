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

//Standard utilities and systems includes
#include <stdio.h>
#include "particles.h"
#include "particles.h"


//Simulation parameters

static const size_t wgSize = 64;

static size_t uSnap(size_t a, size_t b){
  return ((a % b) == 0) ? a : (a - (a % b) + b);
}

void integrateSystem(
    iris::DMem& d_Pos,
    iris::DMem& d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);

  launch("integrateSystemK", globalWorkSize, wgSize, d_Pos, d_Vel, params, deltaTime, numParticles);
}

void calcHash(
    iris::DMem& d_Hash,
    iris::DMem& d_Index,
    iris::DMem& d_Pos,
    const simParams_t &params,
    const int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);

  launch("calcHashK", globalWorkSize, wgSize, d_Hash, d_Index, d_Pos, params, numParticles);
}

void memSet(
    iris::DMem& d_Data,
    unsigned int val,
    unsigned int N)
{
  size_t globalWorkSize = uSnap(N, wgSize);


  launch("memSetK", globalWorkSize, wgSize, d_Data, val, N);
}

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
    const unsigned int numCells)
{
  memSet(d_CellStart, 0xFFFFFFFFU, numCells);

  size_t globalWorkSize = uSnap(numParticles, wgSize);

  launch("findCellBoundsAndReorderK", globalWorkSize, wgSize, d_CellStart,
      d_CellEnd, 
      d_ReorderedPos,
      d_ReorderedVel, 
      d_Hash, 
      d_Index, 
      d_Pos, 
      d_Vel,
      numParticles);
}

void collide(
    iris::DMem& d_Vel,
    iris::DMem& d_ReorderedPos,
    iris::DMem& d_ReorderedVel,
    iris::DMem& d_Index,
    iris::DMem& d_CellStart,
    iris::DMem& d_CellEnd,
    const simParams_t &params,
    const unsigned int   numParticles,
    const unsigned int   numCells)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);


  launch("collideK", globalWorkSize, wgSize, d_Vel, d_ReorderedPos, d_ReorderedVel, 
      d_Index, d_CellStart, d_CellEnd, 
      params, numParticles);
}

