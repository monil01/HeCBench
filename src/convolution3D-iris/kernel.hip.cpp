/*
  Reference
  Chapter 16 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>

#define TILE_WIDTH 16

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

#ifdef MIOPEN_CONV
#error Optional vendor convolution requires an IRIS vendor adapter
#endif

template <typename T>
void verify (const T* Y, T* Y_ref, size_t Y_size)
{
  bool ok = true;
  for (size_t i = 0; i < Y_size; i++) {
    if (fabs(Y[i] - Y_ref[i]) > 1e-3f) {
      printf("%f (device) != %f (reference)\n", Y[i], Y_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template<typename T>
__device__
void conv3d_s1(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int n = blockIdx.x;
  int m = blockIdx.y;
  int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

template<typename T>
__device__
void conv3d_s2(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int m = blockIdx.x;
  int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

template<typename T>
__device__
void conv3d_s3(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  int m = blockIdx.z;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}



extern "C" __global__ void conv3d_1(const float* x,const float* w,float* y,int c,int m,int k,int hi,int wi,int ho,int wo,int wg) { conv3d_s1<float>(x,w,y,c,m,k,hi,wi,ho,wo,wg); }

extern "C" __global__ void conv3d_2(const float* x,const float* w,float* y,int c,int m,int k,int hi,int wi,int ho,int wo,int wg) { conv3d_s2<float>(x,w,y,c,m,k,hi,wi,ho,wo,wg); }

extern "C" __global__ void conv3d_3(const float* x,const float* w,float* y,int c,int m,int k,int hi,int wi,int ho,int wo,int wg) { conv3d_s3<float>(x,w,y,c,m,k,hi,wi,ho,wo,wg); }
