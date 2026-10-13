#include <hip/hip_runtime.h>
#define max(x,y) ((x)>(y)?(x):(y))
extern "C" __global__
void hypterm_1 (double * __restrict__ flux_0,
                double * __restrict__ flux_1,
                double * __restrict__ flux_2,
                double * __restrict__ flux_3,
                double * __restrict__ flux_4,
                const double * __restrict__ cons_1,
                const double * __restrict__ cons_2,
                const double * __restrict__ cons_3,
                const double * __restrict__ cons_4, 
                const double * __restrict__ q_1,
                const double * __restrict__ q_2,
                const double * __restrict__ q_3,
                const double * __restrict__ q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(blockDim.x);
  int i0 = (int)(blockIdx.x)*(blockdim_i);
  int i = max (i0, 0) + (int)(threadIdx.x);
  int blockdim_j= (int)(blockDim.y);
  int j0 = (int)(blockIdx.y)*(blockdim_j);
  int j = max (j0, 0) + (int)(threadIdx.y);
  int blockdim_k= (int)(blockDim.z);
  int k0 = (int)(blockIdx.z)*(blockdim_k);
  int k = max (k0, 0) + (int)(threadIdx.z);

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1])-0.2f*(cons_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2])+0.038f*(cons_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3])-0.0035f*(cons_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_1[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]))-0.2f*(cons_1[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]))+0.038f*(cons_1[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]))-0.0035f*(cons_1[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4])))*dxinv0);
  	 flux_2[k*M*N+j*N+i] = -((0.8f*(cons_2[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_2[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_2[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_2[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_2[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_2[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_2[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_2[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_3[k*M*N+j*N+i] = -((0.8f*(cons_3[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_3[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_3[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_3[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_3[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_3[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_3[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_3[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_4[k*M*N+j*N+i] = -((0.8f*(cons_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]))-0.2f*(cons_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]))+0.038f*(cons_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]))-0.0035f*(cons_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4])))*dxinv0);
  } 
}

extern "C" __global__
void hypterm_2 (double * __restrict__ flux_0,
                double * __restrict__ flux_1,
                double * __restrict__ flux_2,
                double * __restrict__ flux_3,
                double * __restrict__ flux_4,
                const double * __restrict__ cons_1,
                const double * __restrict__ cons_2,
                const double * __restrict__ cons_3,
                const double * __restrict__ cons_4, 
                const double * __restrict__ q_1,
                const double * __restrict__ q_2,
                const double * __restrict__ q_3,
                const double * __restrict__ q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(blockDim.x);
  int i0 = (int)(blockIdx.x)*(blockdim_i);
  int i = max (i0, 0) + (int)(threadIdx.x);
  int blockdim_j= (int)(blockDim.y);
  int j0 = (int)(blockIdx.y)*(blockdim_j);
  int j = max (j0, 0) + (int)(threadIdx.y);
  int blockdim_k= (int)(blockDim.z);
  int k0 = (int)(blockIdx.z)*(blockdim_k);
  int k = max (k0, 0) + (int)(threadIdx.z);

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i])-0.2f*(cons_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i])+0.038f*(cons_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_1[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_1[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_1[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_1[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_1[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_1[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_1[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]))-0.2f*(cons_2[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]))+0.038f*(cons_2[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]))-0.0035f*(cons_2[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i])))*dxinv1;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_3[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_3[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_3[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_3[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_3[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_3[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_3[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]))-0.2f*(cons_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]))+0.038f*(cons_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]))-0.0035f*(cons_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i])))*dxinv2;
  } 
}


extern "C" __global__
void hypterm_3 (double * __restrict__ flux_0,
                double * __restrict__ flux_1,
                double * __restrict__ flux_2,
                double * __restrict__ flux_3,
                double * __restrict__ flux_4,
                const double * __restrict__ cons_1,
                const double * __restrict__ cons_2,
                const double * __restrict__ cons_3,
                const double * __restrict__ cons_4, 
                const double * __restrict__ q_1,
                const double * __restrict__ q_2,
                const double * __restrict__ q_3,
                const double * __restrict__ q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(blockDim.x);
  int i0 = (int)(blockIdx.x)*(blockdim_i);
  int i = max (i0, 0) + (int)(threadIdx.x);
  int blockdim_j= (int)(blockDim.y);
  int j0 = (int)(blockIdx.y)*(blockdim_j);
  int j = max (j0, 0) + (int)(threadIdx.y);
  int blockdim_k= (int)(blockDim.z);
  int k0 = (int)(blockIdx.z)*(blockdim_k);
  int k = max (k0, 0) + (int)(threadIdx.z);

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i])-0.2f*(cons_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i])+0.038f*(cons_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_1[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_1[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_1[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_1[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_1[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_1[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_1[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_2[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_2[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_2[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_2[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_2[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_2[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_2[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]))-0.2f*(cons_3[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]))+0.038f*(cons_3[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]))-0.0035f*(cons_3[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i])))*dxinv2;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]))-0.2f*(cons_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]))+0.038f*(cons_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]))-0.0035f*(cons_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i])))*dxinv1;
  } 
}
