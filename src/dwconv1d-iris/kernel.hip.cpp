#include <hip/hip_runtime.h>
#include <cmath>
#include <stdio.h>

// require T <= Tmax, T % 4 == 0, B % BF == 0, B % BB === 0 (Tmax and BF and BB are passed by compiler)

#define F4(A, B) ((float4 *)(A))[(B) >> 2]

template <typename F>
__device__ void forward_impl(const F *__restrict__ const __w, const F *__restrict__ const __k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;
    const int ti = t + T * i;
    const int tj = T * (B * C) / BF;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax * BF];
    F4(ww, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BF; j++) {
        F4(kk, t + Tmax * j) = F4(__k, ti + tj * j);
    }
    __syncthreads();

    float4 ss[BF];
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        ss[j] = {eps, eps, eps, eps};
    }
    for (int u = 0; u <= t; u++) {
        const F *__restrict__ const w = ww + T - t + u - 4;
        #pragma unroll
        for (int j = 0; j < BF; j++) {
            float4 *__restrict__ const s = ss + j;
            const F k = kk[u + Tmax * j];
            s->x += w[3] * k;
            s->y += w[2] * k;
            s->z += w[1] * k;
            s->w += w[0] * k;
        }
    }
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const w = ww + T - 3;
        const F *__restrict__ const k = kk + Tmax * j + t + 1;
        s->y += w[2] * k[0];
        s->z += w[1] * k[0];
        s->z += w[2] * k[1];
        s->w += w[0] * k[0];
        s->w += w[1] * k[1];
        s->w += w[2] * k[2];
        F4(x, ti + tj * j) = *s;
    }
}

template <typename F>
__device__ void backward_impl(const F *__restrict__ const __w, const F *__restrict__ const __k, const F *__restrict__ const __gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T) {
    const int i = blockIdx.y;
    const int t = threadIdx.x << 2;
    const int ti = t + T * i;
    const int tj = T * (B * C) / BB;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax * BB];
    __shared__ F gg[Tmax * BB];
    F4(ww, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        F4(kk, t + Tmax * j) = F4(__k, ti + tj * j);
        F4(gg, t + Tmax * j) = F4(__gwk, ti + tj * j);
    }
    __syncthreads();

    float4 ss[BB];
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        ss[j] = {0, 0, 0, 0};
    }
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            float4 *__restrict__ const s = ss + j;
            const F *__restrict__ const g = gg + Tmax * j + T - t + u - 4;
            const F k = kk[u + Tmax * j];
            s->x += g[3] * k;
            s->y += g[2] * k;
            s->z += g[1] * k;
            s->w += g[0] * k;
        }
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const k = kk + Tmax * j + t + 1;
        const F *__restrict__ const g = gg + Tmax * j + T - 3;
        s->y += g[2] * k[0];
        s->z += g[1] * k[0];
        s->z += g[2] * k[1];
        s->w += g[0] * k[0];
        s->w += g[1] * k[1];
        s->w += g[2] * k[2];
        F4(gw, ti + tj * j) = *s;
    }

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        ss[j] = {0, 0, 0, 0};
    }
    for (int u = t + 3; u < T; u++) {
        const F w = ww[u];
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            float4 *__restrict__ const s = ss + j;
            const F *__restrict__ const g = gg + Tmax * j + T + t - u - 1;
            s->x += g[0] * w;
            s->y += g[1] * w;
            s->z += g[2] * w;
            s->w += g[3] * w;
        }
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const g = gg + Tmax * j + T - 3;
        const F *__restrict__ const w = ww + t;
        s->x += g[2] * w[0];
        s->x += g[1] * w[1];
        s->x += g[0] * w[2];
        s->y += g[2] * w[1];
        s->y += g[1] * w[2];
        s->z += g[2] * w[2];
        F4(gk, ti + tj * j) = *s;
    }
}


extern "C" __global__ void forward(const float*w,const float*k,float*x,float eps,int B,int C,int T){forward_impl<float>(w,k,x,eps,B,C,T);}
extern "C" __global__ void backward(const float*w,const float*k,const float*go,float*gw,float*gk,int B,int C,int T){backward_impl<float>(w,k,go,gw,gk,B,C,T);}
extern "C" __global__ void loss_grad(const float*x,float*g,float*values,int n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float t=tanhf(x[i]);g[i]=2*x[i]-(1-t*t);values[i]=x[i]*x[i]-t;}}
extern "C" __global__ void reduce_gw(const float*gw,float*result,int B,int C,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<C*T){float s=0;for(int b=0;b<B;++b)s+=gw[b*C*T+i];result[i]=s;}}
extern "C" __global__ void reduce_loss(const float*values,float*result,int n){if(blockIdx.x||threadIdx.x)return;double s=0;for(int i=0;i<n;++i)s+=values[i];result[0]=s;}
