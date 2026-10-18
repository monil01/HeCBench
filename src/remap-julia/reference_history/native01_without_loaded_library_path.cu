#define main historical_cli_main
#include "../remap-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
extern "C" int remap_sequence(int*,int*,int,void*);
extern "C" int remap_sort(int*,int*,int,void*);
extern "C" int remap_unique(int*,int*,int,int*,void*);
extern "C" int remap_thrust_version();
extern "C" int remap_runtime_version();
constexpr int G=64,SENT=-123456789;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
void vk(int e){if(e){fprintf(stderr,"Vendor status %d\n",e);exit(3);}}
std::vector<int>load(const char*n){std::ifstream in(std::string(REMAP_ROOT)+"/"+n,std::ios::binary|std::ios::ate);if(!in)exit(4);size_t bytes=in.tellg();if(bytes%4)exit(5);std::vector<int>v(bytes/4);in.seekg(0);if(!in.read((char*)v.data(),bytes))exit(6);return v;}
std::vector<int>get(int*p,size_t n){std::vector<int>v(n);ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDefault));return v;}
struct Case{int n;size_t prefix_pos;std::vector<int>input;int*base[4];int*p[4];};
void reset(Case&c){std::vector<int>v(c.n+2*G,SENT);for(int i=0;i<4;i++)ck(cudaMemcpy(c.base[i],v.data(),v.size()*4,cudaMemcpyDefault));ck(cudaMemcpy(c.p[0],c.input.data(),c.n*4,cudaMemcpyDefault));ck(cudaDeviceSynchronize());}
void append(std::vector<int>&o,const std::vector<int>&v){o.insert(o.end(),v.begin(),v.end());}
std::vector<int>cycle(Case&c,cudaStream_t stream,std::vector<double>&times,std::vector<int>&permutation,bool&ok){
 auto timed=[&](int stage,auto fn){ck(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();fn();ck(cudaDeviceSynchronize());times[stage]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();};
 timed(0,[&]{vk(remap_sequence(c.p[1],c.p[2],c.n,stream));});
 timed(1,[&]{vk(remap_sort(c.p[0],c.p[1],c.n,stream));});
 auto sorted=get(c.p[0],c.n);permutation=get(c.p[1],c.n);auto checkperm=permutation;std::sort(checkperm.begin(),checkperm.end());for(int i=0;i<c.n;i++)ok &=checkperm[i]==i && permutation[i]>=0 && permutation[i]<c.n;
 if(ok)for(int i=0;i<c.n;i++)ok &=c.input[permutation[i]]==sorted[i];
 int k=0;timed(2,[&]{vk(remap_unique(c.p[0],c.p[2],c.n,&k,stream));});
 if(k<1 || k>c.n)exit(7);
 timed(3,[&]{remap_kernel<<<(k+255)/256,256,0,stream>>>(thrust::device_pointer_cast((const int*)c.p[2]),thrust::device_pointer_cast((const int*)c.p[1]),c.p[3],c.n,k);ck(cudaGetLastError());});
 auto out=sorted;out.push_back(k);append(out,get(c.p[0],k));append(out,get(c.p[2],k));append(out,get(c.p[3],c.n));
 ok &=get(c.p[1],c.n)==permutation;
 for(int i=0;i<4;i++){append(out,get(c.base[i],G));append(out,get(c.p[i]+c.n,G));}
 return out;
}
int main(){auto f=load("fixtures_i32.bin"),oracle=load("oracle_i32.bin");size_t pos=1;std::vector<Case>cases;for(int ci=0;ci<f[0];ci++){Case c;c.n=f.at(pos++);c.prefix_pos=pos;c.input.assign(f.begin()+pos,f.begin()+pos+c.n);pos+=c.n;for(int i=0;i<4;i++){ck(cudaMalloc(&c.base[i],(c.n+2*G)*4));c.p[i]=c.base[i]+G;}cases.push_back(c);}if(pos!=f.size())return 8;cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));printf("VENDOR thrust=%d runtime=%d stream=nonblocking completion=context\n",remap_thrust_version(),remap_runtime_version());
 for(auto&c:cases){reset(c);std::vector<double>t(4);std::vector<int>p;bool ok=true;cycle(c,stream,t,p,ok);if(!ok)return 9;}
 for(int trial=1;trial<=5;trial++){auto payload=f;std::vector<int>results;std::vector<double>times(4);std::vector<std::vector<int>>perms;bool ok=true;for(auto&c:cases){reset(c);auto actual=get(c.p[0],c.n);std::copy(actual.begin(),actual.end(),payload.begin()+c.prefix_pos);std::vector<int>p;append(results,cycle(c,stream,times,p,ok));perms.push_back(p);}ok &=payload==f && results==oracle;append(payload,results);double total=0;for(double t:times)total+=t;std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)payload.data(),payload.size()*4);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":null,\"timing_region\":\"sum8_context_completed_sequence_sort_unique_authored_remap_stages\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_authored_kernel_calls\":8,\"stage_times_seconds\":[",trial,total,ok?"true":"false",payload.size(),ok?0:1);for(int i=0;i<4;i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"permutation_witness\":[");for(size_t c=0;c<perms.size();c++){printf("%s[",c?",":"");for(size_t j=0;j<perms[c].size();j++)printf("%s%d",j?",":"",perms[c][j]);printf("]");}printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok)return 10;}
 ck(cudaStreamDestroy(stream));for(auto&c:cases)for(int i=0;i<4;i++)ck(cudaFree(c.base[i]));return 0;}
