/*
MPC code [double] (LnVs BIT LVs ZE): A GPU-based compressor for arrays of
double-precision floating-point values.  See the following publication for
more information: http://cs.txstate.edu/~mb92/papers/cluster15.pdf.

Copyright (c) 2015-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Annie Yang and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/MPC/.

Publication: This work is described in detail in the following paper.
Annie Yang, Hari Mukka, Farbod Hesaaraki, and Martin Burtscher. MPC: A
Massively Parallel Compression Algorithm for Scientific Data. Proceedings
of the IEEE International Conference on Cluster Computing, pp. 381-389.
September 2015.
*/

#include <cstdio>
#include <cassert>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>
#include "utils.h"

#define TPB 1024  /* do not change */



#include "dmem.hpp"
int main(int argc, char *argv[])
{
  port::Session session(&argc, &argv);
  printf("MPC - Massively Parallel Compression [double] (%s)\n", __FILE__);
  printf("Copyright 2015-2020 Texas State University\n\n");

  assert(sizeof(long) == sizeof(double));

  if ((argc != 2) && (argc != 3)) {
    printf("usage: %s file [dimension]\n", argv[0]);
    exit(-1);
  }

  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);

  if ((deviceProp.warpSize != 32) && (deviceProp.warpSize != 64)) {
    printf("Only a warp size of 32 or 64 is supported. Exit..\n");
    exit(-1);
  }

  const int blocks = deviceProp.multiProcessorCount * 2;

  int dim, insize, outsize;

  std::string name = argv[1];
  long* const input = readFile(name.c_str(), insize);

  if (argc == 3) {
    dim = atoi(argv[2]);
    // Bit-plane padding can emit one reset delta beyond the last bitmap group.
    outsize = ((insize + 63) / 64) * 64 + 2 + (insize + 63) / 64;
  } else {
    assert(((input[0] >> 8) & 0xffffff) == 0x43504d);
    dim = (input[0] & 31) + 1;
    outsize = input[0] >> 32;
  }
  assert(0 < dim);  assert(dim <= 32);
  long* const output = new long[outsize];

  long *d_in, *d_out;
  int *d_offs;
  port::allocate(&d_in, insize * sizeof(long));
  port::allocate(&d_out, outsize * sizeof(long));
  port::allocate(&d_offs, blocks * sizeof(int));
  port::copy(d_in, input, insize * sizeof(long), port::h2d);

  if (argc == 3) {
    auto start = std::chrono::steady_clock::now();
    port::zero(d_offs, -1, blocks * sizeof(int));
    if (deviceProp.warpSize == 64)
      port::launch("MPCcompress64", blocks, TPB, insize, d_in, d_out, d_offs, dim);
    else
      port::launch("MPCcompress32", blocks, TPB, insize, d_in, d_out, d_offs, dim);
    iris_synchronize();
    auto end = std::chrono::steady_clock::now();

    port::copy(output, d_out, sizeof(long), port::d2h);
    outsize = output[0] >> 32;

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("compression time: %.2f ms\n", time * 1e-6);
    printf("compression throughput: %.3f GB/s\n", 1.0 * sizeof(long) * insize / time);
    printf("compression ratio: %.3f\n\n", 1.0 * insize / outsize);

    port::copy(output, d_out, outsize * sizeof(long), port::d2h);
    output[0] = (((long)insize) << 32) + (0x43504d00 - 1) + dim;

    name = "compression.txt";

  } else {

    auto start = std::chrono::steady_clock::now();
    port::zero(d_offs, -1, blocks * sizeof(int));
    if (deviceProp.warpSize == 64)
      port::launch("MPCdecompress64", blocks, TPB, d_in, d_out, d_offs);
    else
      port::launch("MPCdecompress32", blocks, TPB, d_in, d_out, d_offs);
    iris_synchronize();
    auto end = std::chrono::steady_clock::now();

    port::copy(output, d_out, outsize * sizeof(long), port::d2h);

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("decompression time: %.2f ms\n", time * 1e-6);
    printf("decompression throughput: %.3f GB/s\n\n", 1.0 * sizeof(long) * outsize / time);

    name = "decompression.txt";
  }

  writeFile(name.c_str(), output, outsize);

  delete [] output;
  delete [] input;

  port::release(d_offs);
  port::release(d_out);
  port::release(d_in);

  return 0;
}
