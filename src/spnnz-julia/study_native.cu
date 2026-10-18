
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dlfcn.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstring>
#include "study_cases.h"
#define CK(x) do{if((x)!=0)throw std::runtime_error(#x);}while(0)
const unsigned G=16, GUARD=0xa5a55a5a, POISON=0x13579bdf;const int R=10;
std::vector<unsigned>readv(const char*n){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixtureopen");size_t sz=f.tellg();if(sz%4)throw std::runtime_error("dtype");f.seekg(0);std::vector<unsigned>v(sz/4);if(!f.read((char*)v.data(),sz))throw std::runtime_error("shortfixture");return v;}
unsigned*upload(const std::vector<unsigned>&v){unsigned*p;CK(cudaMalloc(&p,v.size()*4));CK(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));return p;}
std::vector<unsigned>get(unsigned*p,int n){std::vector<unsigned>v(n);CK(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
struct State{Case c;std::vector<unsigned>a,y;unsigned*da,*dy;};
int main(){auto input=readv("input_bits_u32.bin"),oracle=readv("oracle_counts_u32.bin");
void*lib=dlopen("/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());auto fn=(decltype(&cusparseSnnz))dlsym(lib,"cusparseSnnz");if(!fn)throw std::runtime_error("missingSnnz");Dl_info info;CK(dladdr((void*)fn,&info)==0);printf("STUDY_VENDOR function=cusparseSnnz path=%s\n",info.dli_fname);
cusparseHandle_t handle;cusparseMatDescr_t desc;CK(cusparseCreate(&handle));CK(cusparseCreateMatDescr(&desc));CK(cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL));CK(cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO));int version;CK(cusparseGetVersion(handle,&version));cusparsePointerMode_t pm;CK(cusparseGetPointerMode(handle,&pm));cudaStream_t stream;CK(cusparseGetStream(handle,&stream));if(pm!=CUSPARSE_POINTER_MODE_HOST||stream!=0)throw std::runtime_error("vendorpolicy");printf("STUDY_VENDOR version=%d pointer_mode=%d stream=0\n",version,int(pm));
std::vector<State>states;std::vector<unsigned>prefix;
for(auto c:cases){State s;s.c=c;s.a.assign(c.n*c.n+32,GUARD);std::copy(input.begin()+c.io,input.begin()+c.io+c.n*c.n,s.a.begin()+G);s.y.assign(c.n+32,GUARD);std::fill(s.y.begin()+G,s.y.end()-G,POISON);s.da=upload(s.a);s.dy=upload(s.y);auto a=get(s.da,s.a.size()),y=get(s.dy,s.y.size());prefix.insert(prefix.end(),a.begin(),a.end());prefix.insert(prefix.end(),y.begin(),y.end());int total=-1;CK(cudaDeviceSynchronize());CK(fn(handle,CUSPARSE_DIRECTION_COLUMN,c.n,c.n,desc,(float*)(s.da+G),c.n,(int*)(s.dy+G),&total));CK(cudaDeviceSynchronize());states.push_back(s);}
for(int trial=1;trial<=5;trial++){auto art=prefix;bool ok=true;double seconds=0;std::vector<double>components;
for(auto&s:states){auto c=s.c;CK(cudaMemcpy(s.dy,s.y.data(),s.y.size()*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());int total=-1;auto start=std::chrono::steady_clock::now();for(int r=0;r<R;r++)CK(fn(handle,CUSPARSE_DIRECTION_COLUMN,c.n,c.n,desc,(float*)(s.da+G),c.n,(int*)(s.dy+G),&total));CK(cudaDeviceSynchronize());double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R;seconds+=t;components.push_back(t);
auto a=get(s.da,s.a.size()),y=get(s.dy,s.y.size());ok&=a==s.a&&total==c.total;for(int i=0;i<c.n+32;i++)ok&=y[i]==(i>=G&&i<c.n+G?oracle[c.oo+i-G]:GUARD);art.insert(art.end(),a.begin(),a.end());art.insert(art.end(),y.begin(),y.end());art.push_back(total);}
std::string path="spnnz_trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)art.data(),art.size()*4);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_six_legacy_cusparseSnnz_batches_host_total_and_device_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",seconds,path.c_str(),art.size(),art.size());}
for(auto&s:states){CK(cudaFree(s.da));CK(cudaFree(s.dy));}CK(cusparseDestroyMatDescr(desc));CK(cusparseDestroy(handle));}
