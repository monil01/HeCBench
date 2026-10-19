#define main original_hotspot3D_main
#include "../hotspot3D-cuda/3D.cu"
#undef main
#include "../hotspot3D-cuda/3D_helper.cu"
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#ifndef FIXTURE
#define FIXTURE "fixture.bin"
#define ORACLE "oracle.bin"
#endif
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>std::vector<T>readall(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missingasset");auto s=f.tellg();if(s<=0||s%sizeof(T))throw std::runtime_error("Invalidassetsize");std::vector<T>v(s/sizeof(T));f.seekg(0);f.read((char*)v.data(),s);if(!f)throw std::runtime_error("Shortasset");return v;}
int main(){auto fixture=readall<float>(FIXTURE);auto oracle=readall<double>(ORACLE);for(int trial=0;trial<=5;trial++){size_t off=1;std::vector<double>prefix{fixture[0]},computed;double elapsed=0,err=0;bool pass=true;for(int ci=0;ci<int(fixture[0]);ci++){int n=fixture[off],z=fixture[off+1],steps=fixture[off+2],N=n*n*z,L=N+128;float q[8];std::copy_n(fixture.data()+off+3,8,q);prefix.insert(prefix.end(),fixture.begin()+off,fixture.begin()+off+11);off+=11;float *base[3];for(int k=0;k<3;k++){ck(cudaMalloc(&base[k],L*4));ck(cudaMemcpy(base[k],fixture.data()+off,L*4,cudaMemcpyHostToDevice));std::vector<float>got(L);ck(cudaMemcpy(got.data(),base[k],L*4,cudaMemcpyDeviceToHost));for(int i=0;i<L;i++){pass&=got[i]==fixture[off+i];prefix.push_back(got[i]);}off+=L;}
float*a=base[0]+64,*b=base[2]+64,*power=base[1]+64;dim3 block(64,4),grid(n/64,n/4);ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int it=0;it<steps;it++){hotspot3d<<<grid,block>>>(a,power,b,n,n,z,q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7]);std::swap(a,b);}ck(cudaGetLastError());ck(cudaDeviceSynchronize());elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
float*selected=steps%2?a:b;for(float*ptr:{a-64,b-64,selected-64,base[1]}){std::vector<float>v(L);ck(cudaMemcpy(v.data(),ptr,L*4,cudaMemcpyDeviceToHost));for(float x:v)computed.push_back(x);}for(auto ptr:base)ck(cudaFree(ptr));}
if(off!=fixture.size()||computed.size()!=oracle.size())throw std::runtime_error("Countmismatch");for(size_t i=0;i<oracle.size();i++){double e=fabs(computed[i]-oracle[i]);err=std::max(err,e);pass&=std::isfinite(computed[i])&&e<=2e-4+2e-6*fabs(oracle[i]);}if(trial==0){if(!pass)throw std::runtime_error("WarmupOracleMismatch");continue;}prefix.insert(prefix.end(),computed.begin(),computed.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"timing_region\":\"sum4_original_column_kernel_iteration_loops_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_behavior_known_scientific_defect\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,elapsed,pass?"true":"false",prefix.size(),err,path.c_str(),prefix.size());if(!pass)throw std::runtime_error("FullOracleMismatch");} }
