#include <cuda_runtime.h>
#define p_Np 84
#define p_Nfp 28
#define p_Nfaces 4
#define p_Nfields 6
#define BSIZE 96
__device__ cudaTextureObject_t t_Q;
__device__ cudaTextureObject_t t_partQ;
__device__ cudaTextureObject_t t_vgeo;
__device__ cudaTextureObject_t t_DrDsDt;
__device__ cudaTextureObject_t t_LIFT;
__device__ cudaTextureObject_t t_surfinfo;
__global__ void MaxwellsGPU_VOL_Kernel3D(float *g_rhsQ){

  /* fastest */
  __shared__ float s_Q[p_Nfields*BSIZE];
  __shared__ float s_facs[12];

  const int n = threadIdx.x;
  const int k = blockIdx.x;
  
  /* "coalesced"  */
  int m = n+k*p_Nfields*BSIZE;
  int id = n;
  s_Q[id] = tex1Dfetch<float>(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch<float>(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch<float>(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch<float>(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch<float>(t_Q, m); m+=BSIZE; id+=BSIZE;
  s_Q[id] = tex1Dfetch<float>(t_Q, m); 

  if(p_Np<12 && n==0)
    for(m=0;m<12;++m)
      s_facs[m] = tex1Dfetch<float>(t_vgeo, 12*k+m);
  else if(n<12 && p_Np>=12)
    s_facs[n] = tex1Dfetch<float>(t_vgeo, 12*k+n);

  __syncthreads();

  float dHxdr=0,dHxds=0,dHxdt=0;
  float dHydr=0,dHyds=0,dHydt=0;
  float dHzdr=0,dHzds=0,dHzdt=0;
  float dExdr=0,dExds=0,dExdt=0;
  float dEydr=0,dEyds=0,dEydt=0;
  float dEzdr=0,dEzds=0,dEzdt=0;
  float Q;

  for(m=0;p_Np-m;){
    float4 D = tex1Dfetch<float4>(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;
#if ( (p_Np) % 2 )==0
    D = tex1Dfetch<float4>(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;

#if ( (p_Np)%3 )==0
    D = tex1Dfetch<float4>(t_DrDsDt, n+m*BSIZE);

    id = m;
    Q = s_Q[id]; dHxdr += D.x*Q; dHxds += D.y*Q; dHxdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHydr += D.x*Q; dHyds += D.y*Q; dHydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dHzdr += D.x*Q; dHzds += D.y*Q; dHzdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dExdr += D.x*Q; dExds += D.y*Q; dExdt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEydr += D.x*Q; dEyds += D.y*Q; dEydt += D.z*Q; id += BSIZE;
    Q = s_Q[id]; dEzdr += D.x*Q; dEzds += D.y*Q; dEzdt += D.z*Q; 

    ++m;
#endif
#endif
  }
  
  const float drdx= s_facs[0];
  const float drdy= s_facs[1];
  const float drdz= s_facs[2];
  const float dsdx= s_facs[4];
  const float dsdy= s_facs[5];
  const float dsdz= s_facs[6];
  const float dtdx= s_facs[8];
  const float dtdy= s_facs[9];
  const float dtdz= s_facs[10];
  
  m = n+p_Nfields*BSIZE*k;

  g_rhsQ[m] = -(drdy*dEzdr+dsdy*dEzds+dtdy*dEzdt - drdz*dEydr-dsdz*dEyds-dtdz*dEydt); m += BSIZE;
  g_rhsQ[m] = -(drdz*dExdr+dsdz*dExds+dtdz*dExdt - drdx*dEzdr-dsdx*dEzds-dtdx*dEzdt); m += BSIZE;
  g_rhsQ[m] = -(drdx*dEydr+dsdx*dEyds+dtdx*dEydt - drdy*dExdr-dsdy*dExds-dtdy*dExdt); m += BSIZE;
  g_rhsQ[m] =  (drdy*dHzdr+dsdy*dHzds+dtdy*dHzdt - drdz*dHydr-dsdz*dHyds-dtdz*dHydt); m += BSIZE;
  g_rhsQ[m] =  (drdz*dHxdr+dsdz*dHxds+dtdz*dHxdt - drdx*dHzdr-dsdx*dHzds-dtdx*dHzdt); m += BSIZE;
  g_rhsQ[m] =  (drdx*dHydr+dsdx*dHyds+dtdx*dHydt - drdy*dHxdr-dsdy*dHxds-dtdy*dHxdt); 
}

__global__ void MaxwellsGPU_SURF_Kernel3D(float *g_rhsQ)
{

  __shared__ float s_fluxQ[p_Nfields*p_Nfp*p_Nfaces];

  const int n = threadIdx.x;
  const int k = blockIdx.x;
  int m;

  /* grab surface nodes and store flux in shared memory */
  if(n< (p_Nfp*p_Nfaces) ){
    /* coalesced reads (maybe) */
    m = 7*(k*p_Nfp*p_Nfaces)+n;
    const  int idM   = tex1Dfetch<float>(t_surfinfo, m); m += p_Nfp*p_Nfaces;
           int idP   = tex1Dfetch<float>(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float Fsc = tex1Dfetch<float>(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float Bsc = tex1Dfetch<float>(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float nx  = tex1Dfetch<float>(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float ny  = tex1Dfetch<float>(t_surfinfo, m); m += p_Nfp*p_Nfaces;
    const  float nz  = tex1Dfetch<float>(t_surfinfo, m);

    /* check if idP<0  */
    double dHx, dHy, dHz, dEx, dEy, dEz;
    if(idP<0){
      idP = p_Nfields*(-1-idP);
      
      dHx = Fsc*(tex1Dfetch<float>(t_partQ, idP+0) - tex1Dfetch<float>(t_Q, idM+0*BSIZE));
      dHy = Fsc*(tex1Dfetch<float>(t_partQ, idP+1) - tex1Dfetch<float>(t_Q, idM+1*BSIZE));
      dHz = Fsc*(tex1Dfetch<float>(t_partQ, idP+2) - tex1Dfetch<float>(t_Q, idM+2*BSIZE));
      
      dEx = Fsc*(tex1Dfetch<float>(t_partQ, idP+3) - tex1Dfetch<float>(t_Q, idM+3*BSIZE));
      dEy = Fsc*(tex1Dfetch<float>(t_partQ, idP+4) - tex1Dfetch<float>(t_Q, idM+4*BSIZE));
      dEz = Fsc*(tex1Dfetch<float>(t_partQ, idP+5) - tex1Dfetch<float>(t_Q, idM+5*BSIZE));
    }
    else{
      dHx = Fsc*(tex1Dfetch<float>(t_Q, idP+0*BSIZE) - tex1Dfetch<float>(t_Q, idM+0*BSIZE));
      dHy = Fsc*(tex1Dfetch<float>(t_Q, idP+1*BSIZE) - tex1Dfetch<float>(t_Q, idM+1*BSIZE));
      dHz = Fsc*(tex1Dfetch<float>(t_Q, idP+2*BSIZE) - tex1Dfetch<float>(t_Q, idM+2*BSIZE));
      
      dEx = Fsc*(Bsc*tex1Dfetch<float>(t_Q, idP+3*BSIZE) - tex1Dfetch<float>(t_Q, idM+3*BSIZE));
      dEy = Fsc*(Bsc*tex1Dfetch<float>(t_Q, idP+4*BSIZE) - tex1Dfetch<float>(t_Q, idM+4*BSIZE));
      dEz = Fsc*(Bsc*tex1Dfetch<float>(t_Q, idP+5*BSIZE) - tex1Dfetch<float>(t_Q, idM+5*BSIZE));
    }

    const double ndotdH = nx*dHx + ny*dHy + nz*dHz;
    const double ndotdE = nx*dEx + ny*dEy + nz*dEz;

    m = n;
    s_fluxQ[m] = -ny*dEz + nz*dEy + dHx - ndotdH*nx; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] = -nz*dEx + nx*dEz + dHy - ndotdH*ny; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] = -nx*dEy + ny*dEx + dHz - ndotdH*nz; m += p_Nfp*p_Nfaces;

    s_fluxQ[m] =  ny*dHz - nz*dHy + dEx - ndotdE*nx; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] =  nz*dHx - nx*dHz + dEy - ndotdE*ny; m += p_Nfp*p_Nfaces;
    s_fluxQ[m] =  nx*dHy - ny*dHx + dEz - ndotdE*nz; 
  }

  /* make sure all element data points are cached */
  __syncthreads();

  if(n< (p_Np))
  {
    float rhsHx = 0, rhsHy = 0, rhsHz = 0;
    float rhsEx = 0, rhsEy = 0, rhsEz = 0;
    
    int sk = n;
    /* can manually unroll to 4 because there are 4 faces */
    for(m=0;p_Nfaces*p_Nfp-m;){
      const float4 L = tex1Dfetch<float4>(t_LIFT, sk); sk+=p_Np;

      /* broadcast */
      int sk1 = m;
      rhsHx += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.x*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.y*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.z*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

      /* broadcast */
      sk1 = m;
      rhsHx += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHy += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsHz += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEx += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEy += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      rhsEz += L.w*s_fluxQ[sk1]; sk1 += p_Nfp*p_Nfaces;
      ++m;

    }
    
    m = n+p_Nfields*k*BSIZE;
    g_rhsQ[m] += rhsHx; m += BSIZE;
    g_rhsQ[m] += rhsHy; m += BSIZE;
    g_rhsQ[m] += rhsHz; m += BSIZE;
    g_rhsQ[m] += rhsEx; m += BSIZE;
    g_rhsQ[m] += rhsEy; m += BSIZE;
    g_rhsQ[m] += rhsEz; m += BSIZE;

  }
}

__global__ void MaxwellsGPU_RK_Kernel3D(
  int Ntotal,
        float *__restrict__ g_resQ,
  const float *__restrict__ g_rhsQ,
        float *__restrict__ g_Q,
  float fa, float fb, float fdt){
  
  int n = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(n<Ntotal){
    float rhs = g_rhsQ[n];
    float res = g_resQ[n];
    res = fa*res + fdt*rhs;
    
    g_resQ[n] = res;
    g_Q[n]    += fb*res;
  }

}

__global__ void partial_get_kernel3d(int Ntotal, int *g_index, float *g_partQ){
  
  int n = blockIdx.x * blockDim.x + threadIdx.x;
    
  if(n<Ntotal)
    g_partQ[n] = tex1Dfetch<float>(t_Q, g_index[n]);
  
}
