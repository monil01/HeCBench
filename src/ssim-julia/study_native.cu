#define main source_ssim_main
#include "../ssim-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include "study_cases.h"
constexpr int G=16;constexpr uint32_t SENT=0x4b123456,POISON=0x7fc12345;
template<class T>std::vector<T>readv(const char*n,size_t expected){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f||size_t(f.tellg())!=expected*sizeof(T))throw std::runtime_error(n);f.seekg(0);std::vector<T>x(expected);if(!f.read((char*)x.data(),expected*sizeof(T)))throw std::runtime_error(n);return x;}
uint32_t bits(float f){uint32_t u;memcpy(&u,&f,4);return u;}
float*upload(const std::vector<uint32_t>&x){float*p;CUDA_CHECK(cudaMalloc(&p,x.size()*4));CUDA_CHECK(cudaMemcpy(p,x.data(),x.size()*4,cudaMemcpyHostToDevice));return p;}
std::vector<uint32_t>get(float*p,size_t n){std::vector<uint32_t>x(n);CUDA_CHECK(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
struct State{Case c;std::vector<uint32_t>a,b,reset;float*A,*B,*O;};
float tile(State&s,const Tile&t){util::trilinear_kernel(compute_ssim<7>,0,0,t.bx,t.by,t.bz,s.A,s.B,vec3i(t.gx,t.gy,t.gz),1.f,343.f/342.f,.01f,.03f,s.O);auto begin=thrust::device_ptr<float>(s.O);return thrust::reduce(begin,begin+t.count,0.f,thrust::plus<float>());}
int main(){auto input=readv<uint32_t>("input_u32.bin",121856);auto oracle=readv<double>("oracle_f64.bin",68247);std::vector<State>states;
 for(auto c:cases){State s;s.c=c;s.a.assign(input.begin()+c.io,input.begin()+c.io+c.gcount);s.b.assign(input.begin()+c.io+c.gcount,input.begin()+c.io+2*c.gcount);s.a.resize(c.gcount+G,SENT);s.b.resize(c.gcount+G,SENT);s.reset.assign(c.cap+G,SENT);s.A=upload(s.a);s.B=upload(s.b);s.O=upload(s.reset);CUDA_CHECK(cudaDeviceSynchronize());for(int ti=c.tbegin;ti<c.tend;ti++)tile(s,tiles[ti]);CUDA_CHECK(cudaDeviceSynchronize());states.push_back(s);}
 for(int trial=1;trial<=5;trial++){std::vector<double>prefix,values,guards,times;bool ok=true;double maxabs=0,maxbudget=0;auto assess=[&](double a,double b){double e=fabs(a-b),budget=2e-5+3e-5*fabs(b);ok&=std::isfinite(a)&&e<=budget;maxabs=std::max(maxabs,e);maxbudget=std::max(maxbudget,e/budget);};
 for(size_t ci=0;ci<states.size();ci++){auto&s=states[ci];auto c=s.c;auto ga=get(s.A,c.gcount),gb=get(s.B,c.gcount);for(auto u:ga)prefix.push_back(double(u));for(auto u:gb)prefix.push_back(double(u));uint32_t params[]={uint32_t(c.dx),uint32_t(c.dy),uint32_t(c.dz),uint32_t(c.bx),uint32_t(c.by),uint32_t(c.bz),uint32_t(c.gcount),7,343,bits(1.f),bits(343.f/342.f),bits(.01f),bits(.03f),uint32_t(ci),3,0};for(auto u:params)prefix.push_back(double(u));
 float total=0;double expectedTotal=0;
 for(int ti=c.tbegin;ti<c.tend;ti++){auto t=tiles[ti];for(auto v:{t.bx,t.by,t.bz,t.gx,t.gy,t.gz,t.ox,t.oy,t.oz,t.count,int(ci),ti-c.tbegin})prefix.push_back(v);auto reset=s.reset;std::fill(reset.begin(),reset.begin()+t.count,POISON);CUDA_CHECK(cudaMemcpy(s.O,reset.data(),reset.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();float reduced=tile(s,t);total+=reduced;CUDA_CHECK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());auto out=get(s.O,c.cap+G);double expectedSum=0;for(int i=0;i<t.count;i++){float v;memcpy(&v,&out[i],4);values.push_back(v);assess(v,oracle[t.oo+i]);expectedSum+=oracle[t.oo+i];}expectedTotal+=expectedSum;values.push_back(reduced);values.push_back(total);assess(reduced,expectedSum);assess(total,expectedTotal);for(int i=t.count;i<c.cap+G;i++){guards.push_back(out[i]);ok&=out[i]==SENT;}}
 float normalized=total/float(c.voxels);values.push_back(normalized);assess(normalized,expectedTotal/c.voxels);auto a=get(s.A,c.gcount+G),b=get(s.B,c.gcount+G);ok&=a==s.a&&b==s.b;for(auto&v:{a,b})for(int i=c.gcount;i<c.gcount+G;i++)guards.push_back(v[i]);}
 auto art=prefix;art.insert(art.end(),values.begin(),values.end());art.insert(art.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)art.data(),art.size()*8);double elapsed=0;for(auto v:times)elapsed+=v;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_budget_error\":%.17g,\"timing_region\":\"sum10tiles_compute_ssim7_GPUreduce_to_host_F32accum_device_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"source_tile_invocations\":10,\"component_time_seconds\":[",trial,elapsed,ok?"true":"false",values.size()+guards.size(),maxabs,maxbudget);for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),art.size());fflush(stdout);if(!ok)return 1;}
 for(auto&s:states){CUDA_CHECK(cudaFree(s.A));CUDA_CHECK(cudaFree(s.B));CUDA_CHECK(cudaFree(s.O));}}
