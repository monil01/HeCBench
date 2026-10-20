#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "reference_history/rewrite.h"
__global__ void Inputs(int*,rw::Cut*,int);
__global__ void Convert(int*,int*,int);
__global__ void CutEnumerate(int*,int*,int*,int*,int*,rw::Cut*,int,int);
static void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){if(argc!=2)return 1;try{
 int n=13,pi=4;std::vector<int>a={0,-1,-1,-1,-1,2,6,10,3,2,3,19,14,24};std::vector<int>b={0,-1,-1,-1,-1,4,8,12,5,7,6,21,23,8};std::vector<int>refs(n+1,10);
 int *da,*db,*dc0,*dc1,*dr;rw::Cut *cuts;ck(cudaMalloc(&da,(n+1)*4));ck(cudaMalloc(&db,(n+1)*4));ck(cudaMalloc(&dc0,(n+1)*4));ck(cudaMalloc(&dc1,(n+1)*4));ck(cudaMalloc(&dr,(n+1)*4));ck(cudaMalloc(&cuts,(n+1)*8*sizeof(rw::Cut)));ck(cudaMemset(cuts,0,(n+1)*8*sizeof(rw::Cut)));ck(cudaMemcpy(da,a.data(),(n+1)*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),(n+1)*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dr,refs.data(),(n+1)*4,cudaMemcpyHostToDevice));
 Convert<<<1,512>>>(da,dc0,n+1);Convert<<<1,512>>>(db,dc1,n+1);Inputs<<<1,512>>>(dr,cuts,pi);
 for(int i=pi+1;i<=n;i++){CutEnumerate<<<1,512>>>(da,db,dc0,dc1,dr,cuts,i-1,1);ck(cudaDeviceSynchronize());}
 std::vector<rw::Cut>out((n+1)*8);ck(cudaMemcpy(out.data(),cuts,out.size()*sizeof(rw::Cut),cudaMemcpyDeviceToHost));FILE*f=fopen(argv[1],"wb");fwrite(&n,4,1,f);fwrite(&pi,4,1,f);fwrite(a.data(),4,a.size(),f);fwrite(b.data(),4,b.size(),f);fwrite(refs.data(),4,refs.size(),f);fwrite(out.data(),sizeof(rw::Cut),out.size(),f);fclose(f);printf("NATIVE_SOURCE_CUT_UNIT_CAPTURE %zuCutObjects\n",out.size());ck(cudaFree(da));ck(cudaFree(db));ck(cudaFree(dc0));ck(cudaFree(dc1));ck(cudaFree(dr));ck(cudaFree(cuts));return 0;
}catch(std::exception&e){fprintf(stderr,"%s\n",e.what());return 2;}}
