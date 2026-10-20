// Source subsystem unit only: explicit defined initialized fixture, not CLI claim.
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>
#include "reference_history/robin_hood.h"
#define private public
#include "reference_history/rewrite.h"
#undef private
static void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void array(FILE*f,const T*p,size_t n){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));fwrite(x.data(),sizeof(T),n,f);}
int main(int argc,char**argv){if(argc!=2)return 1;try{
 rw::CPUSolver cpu;memset(&cpu.lib,0,sizeof(cpu.lib));cpu.ReadLibrary();int n=13,pi=4;
 // Level-sorted equivalent of the13-node earlier unit; literal flags retained.
 std::vector<int>a={0,0,0,0,0,2,6,3,2,3,10,17,22,24};
 std::vector<int>b={0,0,0,0,0,4,8,5,7,6,12,19,21,8};
 std::vector<int>levels={0,0,0,0,0,1,1,1,1,1,2,2,3,4},refs(n+1),phase(n+1);std::vector<int>counts={4,9,11,12,13};
 for(int j=pi+1;j<=n;j++){refs[a[j]/2]++;refs[b[j]/2]++;phase[j]=(phase[a[j]/2]^(a[j]&1))&(phase[b[j]/2]^(b[j]&1));}refs[13]++;
 rw::GPUSolver gpu(n);gpu.CopyLib(cpu.lib);ck(cudaMemset(gpu.selectedCuts,0,(n+1)*sizeof(rw::Cut)));ck(cudaMemset(gpu.bestSubgraph,0,(n+1)*4));
 FILE*f=fopen(argv[1],"wb");fwrite(&n,4,1,f);fwrite(&pi,4,1,f);int len=counts.size();fwrite(&len,4,1,f);fwrite(counts.data(),4,len,f);fwrite(a.data(),4,n+1,f);fwrite(b.data(),4,n+1,f);fwrite(levels.data(),4,n+1,f);fwrite(refs.data(),4,n+1,f);fwrite(&cpu.lib,1,sizeof(cpu.lib),f);
 gpu.EnumerateAndPreEvaluate(levels.data(),counts,n,a.data(),b.data(),refs.data(),false);ck(cudaDeviceSynchronize());
 array(f,gpu.fanin0,n+1);array(f,gpu.fanin1,n+1);array(f,gpu.isComplement0,n+1);array(f,gpu.isComplement1,n+1);array(f,gpu.cuts,(n+1)*8);array(f,gpu.selectedCuts,n+1);array(f,gpu.bestSubgraph,n+1);fclose(f);printf("NATIVE_SOURCE_REWRITE_PRE_EVALUATION_CAPTURE n=%d\n",n);return 0;
}catch(std::exception&e){fprintf(stderr,"%s\n",e.what());return 2;}}
