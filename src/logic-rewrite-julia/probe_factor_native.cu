#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "reference_history/refactor.h"
static void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T*put(const std::vector<T>&v){T*d;ck(cudaMalloc(&d,v.size()*sizeof(T)));ck(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>void save(FILE*f,T*d,size_t n){std::vector<T>v(n);ck(cudaMemcpy(v.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));fwrite(v.data(),sizeof(T),n,f);}
int main(int argc,char**argv){try{
 const int n=256;std::vector<int>cuts(3*n),ranges(n),truthranges(n),links(4*n,-1),lens(2*n,0),next{2*n};std::vector<unsigned>truth(n),neg(n),elem(1,0);std::vector<unsigned long long>table(4*n*8,0);
 for(int j=0;j<n;j++){cuts[3*j]=1;cuts[3*j+1]=2;cuts[3*j+2]=3;ranges[j]=3*(j+1);truthranges[j]=j+1;truth[j]=unsigned(j)*0x01010101u;neg[j]=~truth[j];}
 ck(cudaSetDevice(0));ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize,256*1024*1024));ck(cudaDeviceSetLimit(cudaLimitStackSize,65536));
 auto dc=put(cuts),dr=put(ranges),dtr=put(truthranges),dlinks=put(links),dlens=put(lens),dnext=put(next);auto dt=put(truth),dn=put(neg),de=put(elem);auto dtable=put(table);
 FILE*f=fopen(argv[1],"wb");fwrite(&n,4,1,f);save(f,dc,cuts.size());save(f,dr,ranges.size());save(f,dtr,truthranges.size());save(f,dt,truth.size());save(f,dn,neg.size());save(f,de,elem.size());
 factorFromTruth<<<1,512>>>(dc,dr,dtable,dlinks,dlens,dnext,dt,dn,dtr,de,n);ck(cudaGetLastError());ck(cudaDeviceSynchronize());save(f,dtable,table.size());save(f,dlinks,links.size());save(f,dlens,lens.size());save(f,dnext,next.size());fclose(f);puts("UNIT_ORIGINAL_FACTOR_ALL256_THREE_INPUT_FUNCTIONS_CAPTURED");
 }catch(std::exception&e){fprintf(stderr,"factorunit %s\n",e.what());return 1;}}
