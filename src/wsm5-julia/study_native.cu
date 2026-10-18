#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <string>
#include <cstdlib>
#define MKX 4
#include "study_native_kernel.h"
#define CK(x) do { auto e=(x); if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }while(0)
struct Case{unsigned nx,ny;float dt;std::vector<std::vector<float>>a;std::vector<float*>raw,d;};
void launch(Case&c){wsm<<<dim3((c.nx+7)/8,(c.ny+7)/8),dim3(8,8)>>>(c.d[0],c.d[1],c.d[2],c.d[3],c.d[4],c.d[5],c.d[6],c.d[7],c.d[8],c.d[9],c.d[10],c.d[11],c.d[12],c.d[13],c.d[14],c.dt,1,c.nx,1,c.ny,1,3,1,c.nx,1,c.ny,1,3,1,c.nx,1,c.ny,1,3);CK(cudaGetLastError());}
void reset(Case&c){for(int f=0;f<15;f++){std::vector<float>v(c.a[f].size()+128,-777.f);std::copy(c.a[f].begin(),c.a[f].end(),v.begin()+64);CK(cudaMemcpy(c.raw[f],v.data(),v.size()*4,cudaMemcpyHostToDevice));}}
void values(std::ofstream&o,const std::vector<float>&v){for(float x:v){double y=x;o.write((char*)&y,8);}}
void number(std::ofstream&o,double x){o.write((char*)&x,8);}
int main(){try{std::ifstream f(std::string(STUDY_ROOT)+"/fixtures_f32.bin",std::ios::binary);unsigned nc;f.read((char*)&nc,4);if(nc!=3)throw std::runtime_error("case count");std::vector<Case>cs;for(unsigned ci=0;ci<nc;ci++){Case c;f.read((char*)&c.nx,4);f.read((char*)&c.ny,4);f.read((char*)&c.dt,4);for(int k=0;k<15;k++){size_t n=c.nx*c.ny*(k<10?3:1);c.a.emplace_back(n);f.read((char*)c.a.back().data(),n*4);float *d;CK(cudaMalloc(&d,(n+128)*4));c.raw.push_back(d);c.d.push_back(d+64);}cs.push_back(c);}if(!f)throw std::runtime_error("fixture read");
for(auto&c:cs){reset(c);launch(c);CK(cudaDeviceSynchronize());}
for(int trial=1;trial<=5;trial++){std::string path="trial"+std::to_string(trial)+".bin",review="trial"+std::to_string(trial)+"_physics.json";std::ofstream out(path,std::ios::binary);number(out,3);for(auto&c:cs){reset(c);CK(cudaDeviceSynchronize());number(out,c.nx);number(out,c.ny);number(out,c.dt);for(int k=0;k<15;k++){std::vector<float>v(c.a[k].size());CK(cudaMemcpy(v.data(),c.d[k],v.size()*4,cudaMemcpyDeviceToHost));values(out,v);}}
double sec=0;for(auto&c:cs){CK(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();launch(c);CK(cudaDeviceSynchronize());sec+=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count();for(int k=0;k<15;k++){size_t n=c.a[k].size();std::vector<float>v(n+128);CK(cudaMemcpy(v.data(),c.raw[k],v.size()*4,cudaMemcpyDeviceToHost));values(out,std::vector<float>(v.begin()+64,v.begin()+64+n));values(out,std::vector<float>(v.begin(),v.begin()+64));values(out,std::vector<float>(v.begin()+64+n,v.end()));}}out.close();std::string cmd="python3 "+std::string(STUDY_ROOT)+"/study_check.py "+path+" "+review;if(std::system(cmd.c_str())!=0)throw std::runtime_error("independent/sourceCPU check failed");double metrics[3];std::ifstream m(review+".metrics",std::ios::binary);m.read((char*)metrics,24);if(!m)throw std::runtime_error("metrics read");std::cout.precision(17);std::cout<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<sec<<",\"correct\":true,\"checked_values\":212480,\"max_abs_error\":"<<metrics[0]<<",\"max_rel_error\":"<<metrics[1]<<",\"timing_region\":\"sum3_source_kernel_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float64\",\"count\":212480,\"byte_order\":\"little\"},\"physics_review\":\""<<review<<"\"}\n";}
for(auto&c:cs)for(auto d:c.raw)CK(cudaFree(d));return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
