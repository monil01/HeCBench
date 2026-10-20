#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <cuda_runtime.h>
#include "reference_history/robin_hood.h"
#define private public
#include "reference_history/rewrite.h"
#undef private
#include "launch_bridge.h"
LRCallback lr_callback=nullptr;
extern __managed__ int P;
extern __managed__ int N;
static void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
extern "C" void*lr_symbol(int id) noexcept {
 void*p=id==0?static_cast<void*>(&P):static_cast<void*>(&N);cudaPointerAttributes a{};
 cudaError_t e=cudaPointerGetAttributes(&a,p);
 if(e!=cudaSuccess||a.type!=cudaMemoryTypeManaged){fprintf(stderr,"managedpointer lookup failed %d\n",int(e));return nullptr;}
 return a.devicePointer;
}
extern "C" int lr_rewrite_unit(const char*input,const char*output,LRCallback callback){try{
 FILE*f=fopen(input,"rb");int n,pi,len;fread(&n,4,1,f);fread(&pi,4,1,f);fread(&len,4,1,f);std::vector<int>counts(len),a(5*(n+1)),b(a.size()),levels(n+1),refs(n+1),phase(n+1),replace(n+1,-1);fread(counts.data(),4,len,f);fread(a.data(),4,n+1,f);fread(b.data(),4,n+1,f);fread(levels.data(),4,n+1,f);fread(refs.data(),4,n+1,f);rw::Library lib{};fread(&lib,1,sizeof(lib),f);fclose(f);
 for(int j=pi+1;j<=n;j++)phase[j]=(phase[a[j]/2]^(a[j]&1))&(phase[b[j]/2]^(b[j]&1));
 rw::GPUSolver gpu(n);gpu.CopyLib(lib);int cap=int(2.3*(n+1)*4)/4;
 for(int*p:{gpu.fanin0,gpu.fanin1,gpu.isComplement0,gpu.isComplement1,gpu.phase,gpu.replace})ck(cudaMemset(p,0,cap*4));
 ck(cudaMemset(gpu.selectedCuts,0,(n+1)*sizeof(rw::Cut)));ck(cudaMemset(gpu.bestSubgraph,0,(n+1)*4));
 lr_callback=callback;gpu.EnumerateAndPreEvaluate(levels.data(),counts,n,a.data(),b.data(),refs.data(),false);
 // One-candidate controlled replacement unit: source13 remainsselected, other
 // candidate disabledbeforeReplacement in BOTH paths. This is not fullresyn2.
 unsigned char no=0;ck(cudaMemcpy((char*)gpu.selectedCuts+12*sizeof(rw::Cut)+offsetof(rw::Cut,used),&no,1,cudaMemcpyHostToDevice));
 int newn=gpu.ReplaceSubgraphs(n,a.data(),b.data(),phase.data(),replace.data());lr_callback=nullptr;
 f=fopen(output,"wb");fwrite(&n,4,1,f);fwrite(&newn,4,1,f);fwrite(a.data(),4,newn+1,f);fwrite(b.data(),4,newn+1,f);fwrite(replace.data(),4,n+1,f);fclose(f);printf("SOURCE_REWRITE_CORE_UNIT_COMPLETE old=%d new=%d\n",n,newn);return 0;
}catch(std::exception&e){lr_callback=nullptr;fprintf(stderr,"rewrite unit %s\n",e.what());return 1;}}
extern "C" size_t lr_descriptor_size(){return sizeof(LRLaunch);}
