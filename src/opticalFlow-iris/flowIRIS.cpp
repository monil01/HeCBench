#include "common.h"
#include "flowCUDA.h"
#include "launchers.hpp"
#include <memory>
#include <utility>

void ComputeFlowCUDA(const float *I0,const float *I1,int width,int height,int stride,
                     float alpha,int nLevels,int nWarpIters,int nSolverIters,float *u,float *v) {
  printf("Computing optical flow on GPU...\n");
  using Buffer = std::unique_ptr<FlowBuffer>;
  const size_t count=size_t(stride)*height;
  auto allocate=[&]() { return std::make_unique<FlowBuffer>(count); };
  Buffer tmp=allocate(),du0=allocate(),dv0=allocate(),du1=allocate(),dv1=allocate();
  Buffer Ix=allocate(),Iy=allocate(),Iz=allocate();
  Buffer bu=allocate(),bv=allocate(),nu=allocate(),nv=allocate();
  std::vector<Buffer> pI0(nLevels),pI1(nLevels);
  std::vector<int> pW(nLevels),pH(nLevels),pS(nLevels);
  int level=nLevels-1;
  pI0[level]=std::make_unique<FlowBuffer>(count,I0);
  pI1[level]=std::make_unique<FlowBuffer>(count,I1);
  pW[level]=width; pH[level]=height; pS[level]=stride;
  for(;level>0;--level) {
    const int nw=pW[level]/2,nh=pH[level]/2,ns=iAlignUp(nw);
    pI0[level-1]=std::make_unique<FlowBuffer>(size_t(ns)*nh);
    pI1[level-1]=std::make_unique<FlowBuffer>(size_t(ns)*nh);
    Downscale(*pI0[level],pW[level],pH[level],pS[level],nw,nh,ns,*pI0[level-1]);
    Downscale(*pI1[level],pW[level],pH[level],pS[level],nw,nh,ns,*pI1[level-1]);
    pW[level-1]=nw; pH[level-1]=nh; pS[level-1]=ns;
  }
  bu->zero(); bv->zero();
  for(;level<nLevels;++level) {
    for(int warp=0;warp<nWarpIters;++warp) {
      du0->zero(); dv0->zero(); du1->zero(); dv1->zero();
      WarpImage(*pI1[level],pW[level],pH[level],pS[level],*bu,*bv,*tmp);
      ComputeDerivatives(*pI0[level],*tmp,pW[level],pH[level],pS[level],*Ix,*Iy,*Iz);
      for(int iter=0;iter<nSolverIters;++iter) {
        SolveForUpdate(*du0,*dv0,*Ix,*Iy,*Iz,pW[level],pH[level],pS[level],alpha,*du1,*dv1);
        std::swap(du0,du1); std::swap(dv0,dv1);
      }
      Add(*bu,*du0,pH[level]*pS[level],*bu);
      Add(*bv,*dv0,pH[level]*pS[level],*bv);
    }
    if(level!=nLevels-1) {
      const float scaleX=float(pW[level+1])/pW[level];
      Upscale(*bu,pW[level],pH[level],pS[level],pW[level+1],pH[level+1],pS[level+1],scaleX,*nu);
      const float scaleY=float(pH[level+1])/pH[level];
      Upscale(*bv,pW[level],pH[level],pS[level],pW[level+1],pH[level+1],pS[level+1],scaleY,*nv);
      std::swap(bu,nu); std::swap(bv,nv);
    }
  }
  iris::Task flush;
  iris_check(flush.flush_out(bu->mem)); iris_check(flush.flush_out(bv->mem));
  iris_check(flush.submit(iris_gpu,nullptr,true));
  std::memcpy(u,bu->host.data(),count*sizeof(float));
  std::memcpy(v,bv->host.data(),count*sizeof(float));
}
