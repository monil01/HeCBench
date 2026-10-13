/*------------------------------------------------------------------------------
* Copyright 2015-16: Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC
* Based on John D. McCalpin’s original STREAM benchmark for CPUs
*------------------------------------------------------------------------------
* License:
*  1. You are free to use this program and/or to redistribute
*     this program.
*  2. You are free to modify this program for your own use,
*     including commercial use, subject to the publication
*     restrictions in item 3.
*  3. You are free to publish results obtained from running this
*     program, or from works that you derive from this program,
*     with the following limitations:
*     3a. In order to be referred to as "BabelStream benchmark results",
*         published results must be in conformance to the BabelStream
*         Run Rules published at
*         http://github.com/UoB-HPC/BabelStream/wiki/Run-Rules
*         and incorporated herein by reference.
*         The copyright holders retain the
*         right to determine conformity with the Run Rules.
*     3b. Results based on modified source code or on runs not in
*         accordance with the BabelStream Run Rules must be clearly
*         labelled whenever they are published.  Examples of
*         proper labelling include:
*         "tuned BabelStream benchmark results"
*         "based on a variant of the BabelStream benchmark code"
*         Other comparable, clear and reasonable labelling is
*         acceptable.
*     3c. Submission of results to the BabelStream benchmark web site
*         is encouraged, but not required.
*  4. Use of this program or creation of derived works based on this
*     program constitutes acceptance of these licensing restrictions.
*  5. Absolutely no warranty is expressed or implied.
*/


#define TBSIZE 256
#define SCALAR (0.4)
// Same 256-block, 256-thread strided reduction as CUDA and native HIP.

extern "C" __global__ void init_float(float* a, float* b, float* c, float x, float y, float z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = x; b[i] = y; c[i] = z;
}
extern "C" __global__ void copy_float(const float* a, float* c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; c[i] = a[i];
}
extern "C" __global__ void mul_float(float* b, const float* c) {
  const float scalar = SCALAR;
  int i = blockDim.x * blockIdx.x + threadIdx.x; b[i] = scalar * c[i];
}
extern "C" __global__ void add_float(const float* a, const float* b, float* c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; c[i] = a[i] + b[i];
}
extern "C" __global__ void triad_float(float* a, const float* b, const float* c) {
  const float scalar = SCALAR;
  int i = blockDim.x * blockIdx.x + threadIdx.x; a[i] = b[i] + scalar * c[i];
}
extern "C" __global__ void nstream_float(float* a, const float* b, const float* c) {
  const float scalar = SCALAR;
  int i = blockDim.x * blockIdx.x + threadIdx.x; a[i] += b[i] + scalar * c[i];
}
extern "C" __global__ void dot_float(const float* a, const float* b, float* sum, int n) {
  __shared__ float tb_sum[TBSIZE];
  const size_t local_i = threadIdx.x;
  tb_sum[local_i] = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    tb_sum[local_i] += a[i] * b[i];
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (local_i < offset) tb_sum[local_i] += tb_sum[local_i + offset];
  }
  if (local_i == 0) sum[blockIdx.x] = tb_sum[local_i];
}

extern "C" __global__ void init_double(double* a, double* b, double* c, double x, double y, double z) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] = x; b[i] = y; c[i] = z;
}
extern "C" __global__ void copy_double(const double* a, double* c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; c[i] = a[i];
}
extern "C" __global__ void mul_double(double* b, const double* c) {
  const double scalar = SCALAR;
  int i = blockDim.x * blockIdx.x + threadIdx.x; b[i] = scalar * c[i];
}
extern "C" __global__ void add_double(const double* a, const double* b, double* c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; c[i] = a[i] + b[i];
}
extern "C" __global__ void triad_double(double* a, const double* b, const double* c) {
  const double scalar = SCALAR;
  int i = blockDim.x * blockIdx.x + threadIdx.x; a[i] = b[i] + scalar * c[i];
}
extern "C" __global__ void nstream_double(double* a, const double* b, const double* c) {
  const double scalar = SCALAR;
  int i = blockDim.x * blockIdx.x + threadIdx.x; a[i] += b[i] + scalar * c[i];
}
extern "C" __global__ void dot_double(const double* a, const double* b, double* sum, int n) {
  __shared__ double tb_sum[TBSIZE];
  const size_t local_i = threadIdx.x;
  tb_sum[local_i] = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    tb_sum[local_i] += a[i] * b[i];
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (local_i < offset) tb_sum[local_i] += tb_sum[local_i + offset];
  }
  if (local_i == 0) sum[blockIdx.x] = tb_sum[local_i];
}
