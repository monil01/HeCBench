#include <hip/hip_runtime.h>
#include <math.h>

#include "kernel_common.h"

extern "C" __global__ void distance_float(
    const float *__restrict__ A,
    const float *__restrict__ B,
    int m,
    int n,
    const float *__restrict__ scale_A,
    const float *__restrict__ scale_B,
    float *__restrict__ cross_term)
{
  distance_impl(A, B, m, n, scale_A, scale_B, cross_term);
}

extern "C" __global__ void reduce_cross_term_float(
    float *__restrict__ output,
    const float *__restrict__ cross_term,
    int m,
    int n,
    int nblocks)
{
  reduce_cross_term_impl(output, cross_term, m, n, nblocks);
}

extern "C" __global__ void distance_double(
    const double *__restrict__ A,
    const double *__restrict__ B,
    int m,
    int n,
    const double *__restrict__ scale_A,
    const double *__restrict__ scale_B,
    double *__restrict__ cross_term)
{
  distance_impl(A, B, m, n, scale_A, scale_B, cross_term);
}

extern "C" __global__ void reduce_cross_term_double(
    double *__restrict__ output,
    const double *__restrict__ cross_term,
    int m,
    int n,
    int nblocks)
{
  reduce_cross_term_impl(output, cross_term, m, n, nblocks);
}
