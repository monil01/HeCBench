#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include "../hypterm-cuda/kernels.cu"
#undef max
#undef min
#undef ceil
const double SENT=-8192.25;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>std::vector<T>readvec(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=size_t(f.tellg());std::vector<T>a(n/sizeof(T));f.seekg(0);f.read((char*)a.data(),n);return a;}
struct Case{int n,mode,reps,v;double dx[3];std::vector<std::vector<double>>host;std::vector<double*>raw;std::vector<double>zero;};
void launch(int s,Case&c){auto&a=c.raw;dim3 t(16,4,4),b((c.n+15)/16,(c.n+3)/4,(c.n+3)/4);
#define ARGS a[8]+4,a[9]+4,a[10]+4,a[11]+4,a[12]+4,a[0]+4,a[1]+4,a[2]+4,a[3]+4,a[4]+4,a[5]+4,a[6]+4,a[7]+4,c.dx[0],c.dx[1],c.dx[2],c.n,c.n,c.n
 if(s==0)hypterm_1<<<b,t>>>(ARGS);else if(s==1)hypterm_2<<<b,t>>>(ARGS);else hypterm_3<<<b,t>>>(ARGS);
#undef ARGS
 ck(cudaGetLastError());}
int main(){auto f=readvec<double>(HY_FIXTURE),o=readvec<double>(HY_ORACLE);size_t pos=0;std::vector<Case>cases;while(pos<f.size()){Case c;c.n=int(f[pos++]);c.mode=int(f[pos++]);c.reps=int(f[pos++]);for(int a=0;a<3;a++)c.dx[a]=f[pos++];c.v=c.n*c.n*c.n;c.zero.assign(c.v,0.);for(int a=0;a<13;a++){std::vector<double>h(c.v+8,SENT);if(a<8){std::copy(f.begin()+pos,f.begin()+pos+c.v,h.begin()+4);pos+=c.v;}double*d;ck(cudaMalloc(&d,h.size()*8));ck(cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice));c.host.push_back(h);c.raw.push_back(d);}cases.push_back(c);}
for(auto&c:cases){for(int a=8;a<13;a++)ck(cudaMemcpy(c.raw[a]+4,c.zero.data(),c.v*8,cudaMemcpyHostToDevice));for(int s=0;s<3;s++){launch(s,c);ck(cudaDeviceSynchronize());}}
for(int trial=1;trial<=5;trial++){auto payload=f;bool ok=true;double err=0,elapsed=0;size_t oi=0;std::vector<double>times;for(auto&c:cases){for(int a=8;a<13;a++)ck(cudaMemcpy(c.raw[a],c.host[a].data(),(c.v+8)*8,cudaMemcpyHostToDevice));double ct[3]={0,0,0};for(int rep=0;rep<c.reps;rep++){for(int a=8;a<13;a++)ck(cudaMemcpy(c.raw[a]+4,c.zero.data(),c.v*8,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());for(int s=0;s<3;s++){auto start=std::chrono::steady_clock::now();launch(s,c);ck(cudaDeviceSynchronize());ct[s]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}}for(double t:ct){times.push_back(t/c.reps);elapsed+=t/c.reps;}for(int a=0;a<13;a++){std::vector<double>back(c.v+8);ck(cudaMemcpy(back.data(),c.raw[a],back.size()*8,cudaMemcpyDeviceToHost));if(a<8)ok&=memcmp(back.data(),c.host[a].data(),back.size()*8)==0;else{for(int j=0;j<4;j++)ok&=back[j]==SENT&&back[c.v+4+j]==SENT;for(int t=0;t<c.v;t++){double v=back[t+4],ref=o[oi++],e=fabs(v-ref);err=fmax(err,e);ok&=std::isfinite(v)&&e<=1e-12+1e-10*fabs(ref);int i=t%c.n,j=(t/c.n)%c.n,k=t/(c.n*c.n);if(c.mode==0||i<4||j<4||k<4||i>c.n-5||j>c.n-5||k>c.n-5)ok&=v==0.;}}payload.insert(payload.end(),back.begin(),back.end());}}
std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream z(path,std::ios::binary);z.write((char*)payload.data(),payload.size()*8);z.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum9_hypterm_stage_means_complete\",\"jit_excluded\":true,\"internal_repeats\":10,\"timed_kernel_calls\":90,\"case_stage_times_seconds\":[",trial,elapsed,ok?"true":"false",payload.size(),err);for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok)return 3;}
for(auto&c:cases)for(auto a:c.raw)ck(cudaFree(a));}
