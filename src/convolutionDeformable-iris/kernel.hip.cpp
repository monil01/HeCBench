#include <hip/hip_runtime.h>
#include <cmath>
// Source workload: 3x3, stride/dilation 1, padding 1, NCHW contiguous.
__device__ float get(const float*x,int b,int c,int h,int w,int C,int H,int W){return h>=0&&h<H&&w>=0&&w<W?x[((b*C+c)*H+h)*W+w]:0;}
__device__ float sample(const float*x,int b,int c,float h,float w,int C,int H,int W,float&dh,float&dw){
  dh=dw=0;if(h<=-1||h>=H||w<=-1||w>=W)return 0;
  int iy=floorf(h),ix=floorf(w);float fy=h-iy,fx=w-ix;
  float a=get(x,b,c,iy,ix,C,H,W),v=get(x,b,c,iy,ix+1,C,H,W),d=get(x,b,c,iy+1,ix,C,H,W),e=get(x,b,c,iy+1,ix+1,C,H,W);
  dh=(1-fx)*(d-a)+fx*(e-v);dw=(1-fy)*(v-a)+fy*(e-d);
  return (1-fy)*((1-fx)*a+fx*v)+fy*((1-fx)*d+fx*e);
}
extern "C" __global__ void conv(const float*x,const float*w,const float*b,float*y,int B,int C,int O,int H,int W){
  size_t q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=size_t(B)*O*H*W)return;
  int p=q%(H*W),o=q/(H*W)%O,n=q/(O*H*W);float v=b[o];
  for(int c=0;c<C;++c)for(int k=0;k<9;++k)v+=get(x,n,c,p/W+k/3-1,p%W+k%3-1,C,H,W)*w[(o*C+c)*9+k];y[q]=v;
}
extern "C" __global__ void deform(const float*x,const float*w,const float*b,const float*pred,float*y,int B,int C,int O,int H,int W,int G){
  size_t q=blockIdx.x*blockDim.x+threadIdx.x;int S=H*W,P=G*27;if(q>=size_t(B)*O*S)return;
  int p=q%S,o=q/S%O,n=q/(O*S);float v=b[o];
  for(int c=0;c<C;++c)for(int k=0;k<9;++k){int group=c/(C/G);float dh,dw;
    float h=p/W+k/3-1+pred[(n*P+group*18+2*k)*S+p],a=p%W+k%3-1+pred[(n*P+group*18+2*k+1)*S+p];
    float mask=1/(1+expf(-pred[(n*P+G*18+group*9+k)*S+p]));
    v+=w[(o*C+c)*9+k]*mask*sample(x,n,c,h,a,C,H,W,dh,dw);
  }y[q]=v;
}
extern "C" __global__ void deform_backward(const float*x,const float*w,const float*pred,const float*go,float*gx,float*gw,float*gb,float*gp,int B,int C,int O,int H,int W,int G){
  size_t q=blockIdx.x*blockDim.x+threadIdx.x;int S=H*W,P=G*27;if(q>=size_t(B)*O*S)return;
  int p=q%S,o=q/S%O,n=q/(O*S);float grad=go[q];atomicAdd(gb+o,grad);
  for(int c=0;c<C;++c)for(int k=0;k<9;++k){int group=c/(C/G),wi=(o*C+c)*9+k;
    int hi=(n*P+group*18+2*k)*S+p,xi=hi+S,mi=(n*P+G*18+group*9+k)*S+p;
    float h=p/W+k/3-1+pred[hi],a=p%W+k%3-1+pred[xi],mask=1/(1+expf(-pred[mi])),dh,dw;
    float val=sample(x,n,c,h,a,C,H,W,dh,dw),v=grad*w[wi]*mask;
    atomicAdd(gw+wi,grad*mask*val);atomicAdd(gp+hi,v*dh);atomicAdd(gp+xi,v*dw);atomicAdd(gp+mi,grad*w[wi]*val*mask*(1-mask));
    if(h>-1&&h<H&&a>-1&&a<W){int iy=floorf(h),ix=floorf(a);float fy=h-iy,fx=a-ix;
      for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)if(iy+dy>=0&&iy+dy<H&&ix+dx>=0&&ix+dx<W)atomicAdd(gx+((n*C+c)*H+iy+dy)*W+ix+dx,v*(dy?fy:1-fy)*(dx?fx:1-fx));
    }
  }
}
extern "C" __global__ void conv_backward(const float*x,const float*w,const float*go,float*gx,float*gw,float*gb,int B,int C,int O,int H,int W){
  size_t q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=size_t(B)*O*H*W)return;
  int p=q%(H*W),o=q/(H*W)%O,n=q/(O*H*W);float v=go[q];atomicAdd(gb+o,v);
  for(int c=0;c<C;++c)for(int k=0;k<9;++k){int h=p/W+k/3-1,a=p%W+k%3-1;if(h>=0&&h<H&&a>=0&&a<W){int z=((n*C+c)*H+h)*W+a,wi=(o*C+c)*9+k;atomicAdd(gw+wi,v*x[z]);atomicAdd(gx+z,v*w[wi]);}}
}
extern "C" __global__ void loss(const float*y,const float*target,float*go,float*mean,int n){
  if(blockIdx.x||threadIdx.x)return;double v=0;for(int i=0;i<n;++i){v+=target[i]-y[i];go[i]=-1.0f/n;}*mean=v/n;
}
