#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub=hipcub;
#include <chrono>

#define BLOCK_SIZE 16

extern "C" __global__ 
void ccsd_kernel(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                 const double * __restrict__ f2n,    const double * __restrict__ f2t,
                 const double * __restrict__ f3n,    const double * __restrict__ f3t,
                 const double * __restrict__ f4n,    const double * __restrict__ f4t,
                 const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                 const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                 const double * __restrict__ eorb,   const double eaijk,
                 double * __restrict__ emp4i, double * __restrict__ emp5i,
                 double * __restrict__ emp4k, double * __restrict__ emp5k,
                 const int ncor, const int nocc, const int nvir)
{
  typedef cub::BlockReduce<double, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage t1, t2, t3, t4;

  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;

  double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;

  if (b < nvir && c < nvir) {

    const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

    // nvir < 10000 so this should never overflow
    const int bc = b+c*nvir;
    const int cb = c+b*nvir;

    const double f1nbc = f1n[bc];
    const double f1tbc = f1t[bc];
    const double f1ncb = f1n[cb];
    const double f1tcb = f1t[cb];

    const double f2nbc = f2n[bc];
    const double f2tbc = f2t[bc];
    const double f2ncb = f2n[cb];
    const double f2tcb = f2t[cb];

    const double f3nbc = f3n[bc];
    const double f3tbc = f3t[bc];
    const double f3ncb = f3n[cb];
    const double f3tcb = f3t[cb];

    const double f4nbc = f4n[bc];
    const double f4tbc = f4t[bc];
    const double f4ncb = f4n[cb];
    const double f4tcb = f4t[cb];

    s1 = denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
                      - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
                      + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc);

    s2 =  denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
                      - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
                      + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc);

    const double t1v1b = t1v1[b];
    const double t1v2b = t1v2[b];

    const double dintx1c = dintx1[c];
    const double dintx2c = dintx2[c];
    const double dintc1c = dintc1[c];
    const double dintc2c = dintc2[c];

    s3 = denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                     +(f3nbc+f4tbc+f1ncb)*4)
                     + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2);

    s4 = denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                     +(f3tbc+f4nbc+f1tcb)*4)
                     + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2);
  }
  s1 = BlockReduce(t1).Sum(s1);
  s2 = BlockReduce(t2).Sum(s2);
  s3 = BlockReduce(t3).Sum(s3);
  s4 = BlockReduce(t4).Sum(s4);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(emp4i, s1);
    atomicAdd(emp4k, s2);
    atomicAdd(emp5i, s3);
    atomicAdd(emp5k, s4);
  }
}
