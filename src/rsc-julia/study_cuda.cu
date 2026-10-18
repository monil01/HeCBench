#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <unistd.h>
#include "../rsc-cuda/model_eval.cu"
extern "C" void rsc_fit(float*,flowvector*,int,int*,int,int,float,int*,int);
#ifndef FIXTURE
#error FIXTURE required
#endif
#ifndef ORACLE
#error ORACLE required
#endif
#ifndef RAW_DIR
#error RAW_DIR required
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
constexpr int G=16384;
std::vector<double>load(const std::string&name,size_t count){std::ifstream f(name,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=(std::streamoff)(8*count))exit(3);f.seekg(0);std::vector<double>a(count);f.read((char*)a.data(),8*count);if(!f)exit(3);return a;}
template<class T>struct Buffer{T*base,*data;size_t n;std::vector<T>reset,host;Buffer(size_t count,T guard):n(count),reset(n+16,guard),host(n+16){CK(cudaMalloc(&base,(n+16)*sizeof(T)));data=base+8;}void copyin(){CK(cudaMemcpy(base,reset.data(),reset.size()*sizeof(T),cudaMemcpyHostToDevice));}void read(){CK(cudaMemcpy(host.data(),base,host.size()*sizeof(T),cudaMemcpyDeviceToHost));}};
struct Case{int n,m,e,cpu,blocks,threads;float conv;std::vector<flowvector>flows;std::vector<int>pairs,params_dummy;std::vector<float>params;std::vector<int>hc,ho;int id=0,best=-1,minimum=0;Buffer<float>p;Buffer<int>f,c,o,ident;
Case(int N,int M,int E,float C,int CPU,int B,int T,const std::vector<double>&pre,size_t&pos):n(N),m(M),e(E),cpu(CPU),blocks(B),threads(T),conv(C),flows(n),pairs(2*m),params(4*m),hc(m),ho(m),p(4*m,G),f(4*n,G),c(m,G),o(m,G),ident(1,G){static_assert(sizeof(flowvector)==16);for(int i=0;i<n;++i){flows[i]={(int)pre[pos],(int)pre[pos+1],(int)pre[pos+2],(int)pre[pos+3]};pos+=4;}for(int&v:pairs)v=pre[pos++];const int*fp=(const int*)flows.data();for(int i=0;i<4*n;++i)f.reset[8+i]=fp[i];f.copyin();}
std::vector<double>execute(std::vector<double>&raw,std::vector<double>&stages,double&secs,bool&ok){auto begin=std::chrono::steady_clock::now();auto mark=begin;auto stage=[&](){auto now=std::chrono::steady_clock::now();stages.push_back(std::chrono::duration<double>(now-mark).count());mark=now;};
std::fill(params.begin(),params.end(),0);std::fill(hc.begin(),hc.end(),0);std::fill(ho.begin(),ho.end(),0);id=0;best=-1;minimum=n;
std::copy(params.begin(),params.end(),p.reset.begin()+8);std::copy(hc.begin(),hc.end(),c.reset.begin()+8);std::copy(ho.begin(),ho.end(),o.reset.begin()+8);ident.reset[8]=0;p.copyin();c.copyin();o.copyin();ident.copyin();stage();
rsc_fit(params.data(),flows.data(),n,pairs.data(),m,e,conv,&id,cpu);stage();
CK(cudaMemcpy(p.data,params.data(),4*m*sizeof(float),cudaMemcpyHostToDevice));stage();
RANSAC_kernel_block<<<blocks,threads,4>>>(p.data,(flowvector*)f.data,n,m,e,conv,ident.data,c.data,o.data);CK(cudaDeviceSynchronize());stage();
CK(cudaMemcpy(&id,ident.data,sizeof(int),cudaMemcpyDeviceToHost));if(id<0||id>m)exit(4);CK(cudaMemcpy(hc.data(),c.data,id*sizeof(int),cudaMemcpyDeviceToHost));CK(cudaMemcpy(ho.data(),o.data,id*sizeof(int),cudaMemcpyDeviceToHost));stage();
for(int i=0;i<id;++i)if(ho[i]<minimum){minimum=ho[i];best=hc[i];}stage();secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
p.read();f.read();c.read();o.read();ident.read();std::vector<std::pair<int,int>>sorted;std::vector<bool>seen(m,false);for(int i=0;i<id;++i){int k=c.host[8+i],count=o.host[8+i];ok&=k>=0&&k<m&&count>=0&&count<=n;if(k>=0&&k<m){ok&=!seen[k];seen[k]=true;}sorted.push_back({k,count});}std::sort(sorted.begin(),sorted.end());bool valid=id==0?best==-1&&minimum==n:false;for(auto v:sorted)if(v.first==best&&v.second==minimum)valid=true;ok&=valid;
std::vector<double>out={(double)id,(double)minimum,valid?1.:0.};for(float v:params)out.push_back(v);for(auto v:sorted){out.push_back(v.first);out.push_back(v.second);}out.insert(out.end(),2*(m-id),-1.);
raw.push_back(id);raw.push_back(best);raw.push_back(minimum);raw.push_back(valid?1.:0.);for(float v:params)raw.push_back(v);for(int i=0;i<m;++i){raw.push_back(c.host[8+i]);raw.push_back(o.host[8+i]);if(i>=id)ok&=c.host[8+i]==0&&o.host[8+i]==0;}
for(int i=0;i<4*n;++i){out.push_back(f.host[8+i]);raw.push_back(f.host[8+i]);ok&=f.host[8+i]==f.reset[8+i];}for(int i=0;i<4*m;++i){out.push_back(p.host[8+i]);raw.push_back(p.host[8+i]);ok&=p.host[8+i]==params[i];}
auto guards=[&](auto&b){for(size_t j=0;j<b.host.size();++j)if(j<8||j>=b.n+8){out.push_back(b.host[j]);raw.push_back(b.host[j]);ok&=b.host[j]==G;}};guards(p);guards(f);guards(c);guards(o);guards(ident);ok&=ident.host[8]==id;return out;}
};
int main(){auto pre=load(FIXTURE,3061),gold=load(ORACLE,6756);size_t pos=1;std::vector<Case*>cases;for(int i=0;i<4;++i){int n=pre[pos++],m=pre[pos++],e=pre[pos++];float conv=pre[pos++];int cpu=pre[pos++],b=pre[pos++],t=pre[pos++];cases.push_back(new Case(n,m,e,conv,cpu,b,t,pre,pos));}if(pos!=pre.size())return 3;
for(auto c:cases){std::vector<double>raw,st;double sec;bool ok=true;c->execute(raw,st,sec,ok);if(!ok)return 4;}
std::filesystem::create_directories(RAW_DIR);
for(int trial=1;trial<=5;++trial){auto art=pre;std::vector<double>raw,stages,paths;bool ok=true;for(auto c:cases){double secs;auto vals=c->execute(raw,stages,secs,ok);paths.push_back(secs);art.insert(art.end(),vals.begin(),vals.end());}if(art.size()!=pre.size()+gold.size())return 3;for(size_t i=0;i<gold.size();++i)ok&=art[pre.size()+i]==gold[i];std::string path="output_trial"+std::to_string(trial)+".bin",rp=std::string(RAW_DIR)+"/raw_trial"+std::to_string(trial)+"_pid"+std::to_string(getpid())+".bin";for(auto q:{std::make_pair(path,&art),std::make_pair(rp,&raw)}){std::ofstream f(q.first,std::ios::binary);f.write((char*)q.second->data(),q.second->size()*8);if(!f)return 3;}double total=0;for(double s:paths)total+=s;
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"four_hybrid_RANSAC_passes_reset_fit_copy_evaluate_read_select\",\"jit_excluded\":true,\"internal_repeats\":1,\"numerical_agreement_claim\":false,\"raw_output_identity_claim\":false,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"raw_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu},\"path_times_seconds\":[",trial,total,ok?"true":"false",art.size(),path.c_str(),art.size(),rp.c_str(),raw.size());for(size_t i=0;i<paths.size();++i)printf("%s%.17g",i?",":"",paths[i]);printf("],\"stage_times_seconds\":[");for(size_t i=0;i<stages.size();++i)printf("%s%.17g",i?",":"",stages[i]);printf("]}\n");}
}
