#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include "fixture_definition.h"
#include "../stsg-cuda/Filter.cu"
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDAerror %s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T*alloc(size_t n){T*p;check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));return p;}
int main(){STSGFixture f;int X=f.nx,Y=f.ny,B=f.nb,Z=f.years,N=X*Y,W=(2*f.win+1)*(2*f.win+1);auto raw=alloc<short>(f.raw.size());auto rawqa=alloc<unsigned char>(f.qa.size());check(cudaMemcpy(raw,f.raw.data(),f.raw.size()*2,cudaMemcpyHostToDevice));check(cudaMemcpy(rawqa,f.qa.data(),f.qa.size(),cudaMemcpyHostToDevice));auto nd=alloc<float>(N*B*Z),qa=alloc<float>(N*B*Z),ref=alloc<float>(N*B),res=alloc<float>(N*W*4),r3=alloc<float>(N*W*3),out=alloc<float>(N*B*Z),vin=alloc<float>(N*B);auto res1=alloc<int>(N*B),index=alloc<int>(N*W*3);dim3 block(16,16),grid((X+15)/16,(Y+15)/16);
Short_to_Float<<<grid,block>>>(raw,rawqa,X,Y,B,Z,nd,qa);Generate_NDVI_reference<<<grid,block>>>(f.cosyear,2,nd,qa,X,Y,B,Z,ref,r3,res1);check(cudaMemset(r3,0,N*W*3*4));Compute_d_res<<<grid,block>>>(nd,qa,ref,0,Y,0,0,X,Y,B,Z,f.win,res);STSG_filter<<<grid,block>>>(nd,qa,ref,0,Y,0,0,X,Y,B,Z,f.win,f.sampcorr,f.snow,out,vin,res,r3,index);check(cudaGetLastError());check(cudaDeviceSynchronize());std::vector<float>v(N*B*Z);check(cudaMemcpy(v.data(),out,v.size()*4,cudaMemcpyDeviceToHost));FILE* fp=fopen(".porting-state/porting_logs/stsg-julia-shared/native_readiness_output.bin","wb");if(!fp)return 3;if(fwrite(v.data(),4,v.size(),fp)!=v.size())return 4;fclose(fp);int bad=0;float lo=1e30f,hi=-1e30f;for(float x:v){bad+=!std::isfinite(x);lo=std::min(lo,x);hi=std::max(hi,x);}printf("STSG_NATIVE_READINESS source4stages output_count=%zu nonfinite=%d range=[%.12g,%.12g]\n",v.size(),bad,double(lo),double(hi));for(auto d:{nd,qa,ref,res,r3,out,vin})check(cudaFree(d));check(cudaFree(raw));check(cudaFree(rawqa));check(cudaFree(res1));check(cudaFree(index));return bad!=0;}
