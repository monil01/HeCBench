#include "omp_serial_stub.h"
/*
* Portions Copyright (c) 1993-2015 NVIDIA Corporation.  All rights reserved.
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
* Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
* Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
*
* Sobol Quasi-random Number Generator example
*
* Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
* http://people.maths.ox.ac.uk/~gilesm/
*
* and C code developed by Stephen Joe, University of Waikato, New Zealand
* and Frances Kuo, University of New South Wales, Australia
* http://web.maths.unsw.edu.au/~fkuo/sobol/
*
* For theoretical background see:
*
* P. Bratley and B.L. Fox.
* Implementing Sobol's quasirandom sequence generator
* http://portal.acm.org/citation.cfm?id=42288
* ACM Trans. on Math. Software, 14(1):88-100, 1988
*
* S. Joe and F. Kuo.
* Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
* http://portal.acm.org/citation.cfm?id=641879
* ACM Trans. on Math. Software, 29(1):49-57, 2003
*
*/

#include "sobol.h"
#include "sobol_gpu.h"

#define k_2powneg32 2.3283064E-10F


int _ffs(const int x) {
  for (int i = 0; i < 32; i++)
    if ((x >> i) & 1) return (i+1);
  return 0;
};

double sobolGPU(int repeat, int n_vectors, int n_dimensions, 
                unsigned int *dir, float *out)
{
    const int threadsperblock = 128;

    // This implementation of the generator outputs all the draws for
    // one dimension in a contiguous region of memory, followed by the
    // next dimension and so on.
    // Therefore all threads within a block will be processing different
    // vectors from the same dimension. As a result we want the total
    // number of blocks to be a multiple of the number of dimensions.
    size_t dimGrid_y = n_dimensions;
    size_t dimGrid_x;

    // If the number of dimensions is large then we will set the number
    // of blocks to equal the number of dimensions (i.e. dimGrid.x = 1)
    // but if the number of dimensions is small (e.g. less than four per
    // multiprocessor) then we'll partition the vectors across blocks
    // (as well as threads).
    if (n_dimensions < (4 * 24))
    {
        dimGrid_x = 4 * 24;
    }
    else
    {
        dimGrid_x = 1;
    }

    // Cap the dimGrid.x if the number of vectors is small
    if (dimGrid_x > (unsigned int)(n_vectors / threadsperblock))
    {
        dimGrid_x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }

    // Round up to a power of two, required for the algorithm so that
    // stride is a power of two.
    unsigned int targetDimGridX = dimGrid_x;

    for (dimGrid_x = 1 ; dimGrid_x < targetDimGridX ; dimGrid_x *= 2);

    // Fix the number of threads
    size_t numTeam =  dimGrid_x * dimGrid_y;

    auto start = std::chrono::steady_clock::now();

    // Execute GPU kernel
    // FM-3: kernel was launched with (dimGrid_x*dimGrid_y teams, threadsperblock threads).
    // Serial variant iterates the whole (teamY, teamX, tidX) grid explicitly.
    const unsigned int threadSizeX = threadsperblock;
    unsigned int * const dir_base = dir;
    float * const out_base = out;
    for (int rep = 0; rep < repeat; rep++) {
      for (unsigned int teamY = 0; teamY < dimGrid_y; teamY++) {
        unsigned int *dir_l = dir_base + n_directions * teamY;
        float *out_l = out_base + n_vectors * teamY;
        for (unsigned int teamX = 0; teamX < dimGrid_x; teamX++) {
          unsigned int v[n_directions];
          // "shared memory" init: first n_directions threads copy v[]
          for (unsigned int tidX = 0; tidX < n_directions; tidX++) {
            v[tidX] = dir_l[tidX];
          }
          for (unsigned int tidX = 0; tidX < threadSizeX; tidX++) {
            int i0     = teamX * threadSizeX + tidX;
            int stride = dimGrid_x * threadSizeX;

            unsigned int g = i0 ^ (i0 >> 1);
            unsigned int X = 0;
            unsigned int mask;

            for (int k = 0 ; k < _ffs(stride) - 1 ; k++)
            {
                mask = - (g & 1);
                X ^= mask & v[k];
                g = g >> 1;
            }

            if (i0 < n_vectors)
            {
                out_l[i0] = (float)X * k_2powneg32;
            }

            unsigned int v_log2stridem1 = v[_ffs(stride) - 2];
            unsigned int v_stridemask = stride - 1;

            for (int idx = i0 + stride ; idx < n_vectors ; idx += stride)
            {
                X ^= v_log2stridem1 ^ v[_ffs(~((idx - stride) | v_stridemask)) - 1];
                out_l[idx] = (float)X * k_2powneg32;
            }
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return time;
}
