/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <hip/hip_runtime.h>
extern "C" __global__ void clock_block(long* d_o, long clock_count) {
  long clock_offset = 0;
  for (int i = 0; i < clock_count; ++i) clock_offset += i % 3;
  d_o[0] = clock_offset;
}
// Opt-in diagnostic only. Volatile prevents the arithmetic from being moved
// outside the global GPU wall-clock interval; this is not a performance run.
extern "C" __global__ void clock_block_trace(long* d_o, long clock_count,
                                           unsigned long long* interval) {
  interval[0] = wall_clock64();
  volatile long clock_offset = 0;
  for (int i = 0; i < clock_count; ++i) clock_offset = clock_offset + i % 3;
  d_o[0] = clock_offset;
  interval[1] = wall_clock64();
}
extern "C" __global__ void sum(long* d_clocks, int n) {
  __shared__ long s_clocks[32];
  long my_sum = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) my_sum += d_clocks[i];
  s_clocks[threadIdx.x] = my_sum;
  __syncthreads();
  for (int i = 16; i > 0; i /= 2) {
    if (threadIdx.x < i) s_clocks[threadIdx.x] += s_clocks[threadIdx.x + i];
    __syncthreads();
  }
  if (threadIdx.x == 0) d_clocks[0] = s_clocks[0];
}
