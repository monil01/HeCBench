#include <cuda_runtime.h>
#define main original_tsp_main
#include "../tsp-cuda/main.cu"
#undef main
#undef swap
#undef dist
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
using U=unsigned;constexpr int G=64;constexpr U GUARD=0xdeadbeef;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<U>readwords(const char*name){std::ifstream f(std::string(TSP_ROOT)+"/"+name,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("missing words");auto z=f.tellg();if(z%4)throw std::runtime_error("word size");std::vector<U>v(size_t(z)/4);f.seekg(0);f.read((char*)v.data(),z);if(!f)throw std::runtime_error("word read");return v;}
struct Buffer{U*p;int n;};
Buffer alloc(int n){U*p;ck(cudaMalloc(&p,(n+2*G)*4));std::vector<U>h(n+2*G,GUARD);ck(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return {p,n};}
std::vector<U>capture(Buffer b){std::vector<U>h(b.n+2*G);ck(cudaMemcpy(h.data(),b.p,h.size()*4,cudaMemcpyDeviceToHost));return h;}
__global__void pairdiag(int n,const float*x,const float*y,int*out){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<n*n){int a=q/n,b=q%n;float dx=x[a]-x[b],dy=y[a]-y[b];out[q]=int(sqrtf(dx*dx+dy*dy));}}
struct Case{int n,r,t,stride;std::vector<Buffer>b;};
void reset(Case&c){std::vector<U>g(c.stride*c.r,GUARD);ck(cudaMemcpy(c.b[2].p+G,g.data(),g.size()*4,cudaMemcpyHostToDevice));int zero=0,mx=INT_MAX;ck(cudaMemcpy(c.b[3].p+G,&zero,4,cudaMemcpyHostToDevice));ck(cudaMemcpy(c.b[4].p+G,&mx,4,cudaMemcpyHostToDevice));ck(cudaMemset(c.b[5].p+G,0xff,c.n*c.n*4));ck(cudaDeviceSynchronize());}
void launch(Case&c){TwoOpt<<<c.r,c.t,4*c.t>>>(c.n,(float*)(c.b[0].p+G),(float*)(c.b[1].p+G),(int*)(c.b[2].p+G),(int*)(c.b[3].p+G),(int*)(c.b[4].p+G));ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
void diag(Case&c){pairdiag<<<(c.n*c.n+255)/256,256>>>(c.n,(float*)(c.b[0].p+G),(float*)(c.b[1].p+G),(int*)(c.b[5].p+G));ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
int main(){try{auto data=readwords("fixtures_u32.bin"),oracle=readwords("oracle_u32.bin");size_t idx=1;std::vector<Case>cases;for(int ci=0;ci<2;ci++){Case c;c.n=data[idx++];c.r=data[idx++];c.t=data[idx++];if(c.t!=best_thread_count(c.n))throw std::runtime_error("source thread policy");c.stride=(3*c.n+2+31)/32*32;int sizes[]={c.n,c.n,c.stride*c.r,1,1,c.n*c.n};for(int n:sizes)c.b.push_back(alloc(n));for(int xy=0;xy<2;xy++){ck(cudaMemcpy(c.b[xy].p+G,data.data()+idx,c.n*4,cudaMemcpyHostToDevice));idx+=c.n;}cases.push_back(c);}if(idx!=data.size())throw std::runtime_error("fixture parse");for(auto&c:cases){reset(c);launch(c);diag(c);}for(int trial=1;trial<=5;trial++){std::vector<U>prefix{2},suffix;std::vector<double>times;for(auto&c:cases){reset(c);auto a=std::chrono::steady_clock::now();launch(c);auto z=std::chrono::steady_clock::now();times.push_back(std::chrono::duration<double>(z-a).count());diag(c);std::vector<std::vector<U>>h;for(auto b:c.b)h.push_back(capture(b));prefix.insert(prefix.end(),{U(c.n),U(c.r),U(c.t)});for(int xy=0;xy<2;xy++)prefix.insert(prefix.end(),h[xy].begin()+G,h[xy].end()-G);for(int j:{2,3,4,5})suffix.insert(suffix.end(),h[j].begin()+G,h[j].end()-G);for(auto&v:h){suffix.insert(suffix.end(),v.begin(),v.begin()+G);suffix.insert(suffix.end(),v.end()-G,v.end());}}bool ok=prefix==data&&suffix==oracle;std::vector<U>out=prefix;out.insert(out.end(),suffix.begin(),suffix.end());std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*4);if(!f)throw std::runtime_error("capture write");std::cout.precision(17);std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<times[0]+times[1]<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<(ok?0:1)<<",\"max_rel_error\":null,\"timing_region\":\"sum2_source_tiled_twoopt_kernels_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":2,\"untimed_diagnostic_kernel_calls\":2,\"case_times_seconds\":["<<times[0]<<","<<times[1]<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"uint32\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}\n";if(!ok)throw std::runtime_error("full independent oracle mismatch");}for(auto&c:cases)for(auto b:c.b)ck(cudaFree(b.p));return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
