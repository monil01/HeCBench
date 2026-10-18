#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include "../rng-wallace-cuda/constants.h"
#include "../rng-wallace-cuda/wallace_kernel.h"
#define CK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
template<class T>std::vector<T>load(const std::string&f){std::ifstream s(f,std::ios::binary|std::ios::ate);if(!s)throw std::runtime_error(f);size_t n=s.tellg();s.seekg(0);std::vector<T>a(n/sizeof(T));s.read((char*)a.data(),n);return a;}
int main(int argc,char**argv){int rep=argc>1?atoi(argv[1]):3;std::string file=__FILE__;std::string base=file.substr(0,file.find_last_of('/'));auto input=load<float>(base+"/fixture.bin");auto gold=load<double>(base+"/expected_outputs.bin");auto source=load<float>(base+"/source_cpu_outputs.bin");int nb=input[0],n=input[1],np=n+17;float *pool,*out,*chi;CK(cudaMalloc(&pool,np*4));CK(cudaMalloc(&out,np*4));CK(cudaMalloc(&chi,(nb+17)*4));CK(cudaMemcpy(chi,input.data()+12+np,(nb+17)*4,cudaMemcpyHostToDevice));std::vector<float>empty(np,-12345);double initial[257]={};for(int b=0;b<nb;++b)for(int i=0;i<2048;++i){double v=input[12+b*2048+i];initial[b]+=v*v;}
 auto reset=[&](){CK(cudaMemcpy(pool,input.data()+12,np*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(out,empty.data(),np*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());};
 auto launch=[&](unsigned seed,int steps){for(int k=0;k<steps;++k)rng_wallace<<<nb,256>>>(seed,pool,out,chi);CK(cudaGetLastError());};
 for(int c=0;c<3;++c){reset();unsigned seed=unsigned(input[3+3*c])|(unsigned(input[4+3*c])<<16);launch(seed,input[5+3*c]);CK(cudaDeviceSynchronize());}
 for(int tr=1;tr<=5;++tr){std::vector<float>got;std::vector<double>times;double norm=0;
 for(int c=0;c<3;++c){unsigned seed=unsigned(input[3+3*c])|(unsigned(input[4+3*c])<<16);int steps=input[5+3*c];double elapsed=0;for(int r=0;r<rep;++r){reset();auto st=std::chrono::steady_clock::now();launch(seed,steps);CK(cudaDeviceSynchronize());elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count();}times.push_back(elapsed/rep);size_t off=got.size();got.resize(off+2*np);CK(cudaMemcpy(got.data()+off,pool,np*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data()+off+np,out,np*4,cudaMemcpyDeviceToHost));for(int b=0;b<nb;++b){double energy=0;for(int i=0;i<2048;++i){double v=got[off+b*2048+i];energy+=v*v;}norm=std::max(norm,fabs(energy-initial[b])/initial[b]);}}
 bool ok=got.size()==gold.size()&&norm<=1e-5;double ma=0,mr=0,se=0;for(size_t i=0;i<got.size();++i){double e=fabs(got[i]-gold[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(gold[i]),1e-12));se=std::max(se,fabs(double(got[i])-source[i]));ok&=std::isfinite(got[i])&&(gold[i]==-12345?got[i]==-12345:e<=1e-3)&&fabs(got[i]-source[i])<=1e-3;}
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)input.data(),input.size()*4);o.write((char*)got.data(),got.size()*4);o.close();double total=0;for(double t:times)total+=t;
 std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<tr<<",\"time_seconds\":"<<total<<",\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<got.size()<<",\"max_abs_error\":"<<ma<<",\"max_rel_error\":"<<mr<<",\"source_cpu_max_abs_error\":"<<se<<",\"max_relative_pool_squared_norm_drift\":"<<norm<<",\"timing_region\":\"sum3_stateful_chains_1_7_31_kernel_calls_only\",\"jit_excluded\":true,\"internal_repeats\":"<<rep<<",\"per_path_times_seconds\":[";for(int c=0;c<3;++c)std::cout<<(c?",":"")<<times[c];std::cout<<"],\"output_artifact\":{\"path\":\""<<path<<"\",\"dtype\":\"float32\",\"count\":"<<input.size()+got.size()<<",\"byte_order\":\"little\"}}\n";}
}
