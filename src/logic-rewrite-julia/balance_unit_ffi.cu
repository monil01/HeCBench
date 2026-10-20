#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "configured_balance/balance.h"
#include "launch_bridge.h"
LRCallback lr_callback=nullptr;
static void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
extern "C" int lr_balance_unit(const char*input,const char*output,LRCallback callback,int mode){try{
 size_t heap=0;ck(cudaDeviceGetLimit(&heap,cudaLimitMallocHeapSize));if(heap<256*1024*1024)ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize,256*1024*1024));
 FILE*f=fopen(input,"rb");if(!f)throw std::runtime_error("inputopen");int n,pi,len;if(fread(&n,4,1,f)!=1||fread(&pi,4,1,f)!=1||fread(&len,4,1,f)!=1)throw std::runtime_error("inputheader");
 std::vector<int>counts(len),a(n+1),b(n+1),refs(n+1,0),outs{2*n,2*(n-1),2*(n-2),2*(n-3)};fread(counts.data(),4,len,f);fread(a.data(),4,n+1,f);fread(b.data(),4,n+1,f);fclose(f);
 for(int j=pi+1;j<=n;j++){refs[a[j]>>1]++;refs[b[j]>>1]++;}for(int lit:outs)refs[lit>>1]++;
 int*da,*db,*dr,*dout;ck(cudaMalloc(&da,a.size()*4));ck(cudaMalloc(&db,b.size()*4));ck(cudaMalloc(&dr,refs.size()*4));ck(cudaMalloc(&dout,outs.size()*4));
 ck(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dr,refs.data(),refs.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dout,outs.data(),outs.size()*4,cudaMemcpyHostToDevice));
 lr_callback=callback;auto result=balancePerformV2(n+1,pi,outs.size(),n-pi,da,db,dout,dr,mode);lr_callback=nullptr;ck(cudaDeviceSynchronize());
 int nn=std::get<4>(result)+pi+1,levels=-1;a.assign(nn,0);b.assign(nn,0);refs.assign(nn,0);
 // Source returns unspecified PIfanin storage; only ANDprefix is semantic.
 ck(cudaMemcpy(a.data()+pi+1,std::get<0>(result)+pi+1,(nn-pi-1)*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data()+pi+1,std::get<1>(result)+pi+1,(nn-pi-1)*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(outs.data(),std::get<3>(result),outs.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(refs.data(),std::get<2>(result),nn*4,cudaMemcpyDeviceToHost));
 f=fopen(output,"wb");fwrite(&nn,4,1,f);fwrite(&pi,4,1,f);fwrite(&levels,4,1,f);fwrite(a.data(),4,nn,f);fwrite(b.data(),4,nn,f);fwrite(outs.data(),4,outs.size(),f);fwrite(refs.data(),4,nn,f);fclose(f);
 for(int*p:{da,db,dr,dout,std::get<0>(result),std::get<1>(result),std::get<3>(result),std::get<2>(result)})ck(cudaFree(p));return 0;
 }catch(std::exception&e){lr_callback=nullptr;fprintf(stderr,"balanceunit %s\n",e.what());return 1;}}
