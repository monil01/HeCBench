#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include "../lci-cuda/tables.h"
#ifdef LCI_CORRECTED
#include "native_corrected/kernels.h"
#else
#include "../lci-cuda/kernels.h"
#endif
const double GUARD=12345.5;const int G=16;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<double> readv(const char*n,int k){std::vector<double>v(k);std::ifstream f(std::string("/home/imo/HeCBench_julia/src/lci-julia/")+n,std::ios::binary);f.read((char*)v.data(),k*8);if(!f)throw std::runtime_error("fixture read");return v;}
int main(){auto input=readv("inputs_f64.bin",260),al=readv("alpha_f64.bin",193),gold=readv("oracle_f64.bin",256);std::vector<double>alpha(225,GUARD);std::copy(al.begin(),al.end(),alpha.begin()+G);
#ifdef LCI_CORRECTED
check(cudaMemcpyToSymbol(lci_alpha,alpha.data(),225*8));std::vector<double>back(225);check(cudaMemcpyFromSymbol(back.data(),lci_alpha,225*8));if(back!=alpha)throw std::runtime_error("constant upload");
#endif
 double*c[4],*r[4];std::vector<double>prefix;double times[4]={.2,1,200,17.5};for(int q=0;q<4;q++){std::vector<double>v(97,GUARD);std::copy(input.begin()+q*65,input.begin()+(q+1)*65,v.begin()+G);prefix.insert(prefix.end(),v.begin(),v.end());check(cudaMalloc(&c[q],97*8));check(cudaMalloc(&r[q],97*8));check(cudaMemcpy(c[q],v.data(),97*8,cudaMemcpyHostToDevice));}for(int q=0;q<4;q++)prefix.insert(prefix.end(),97,GUARD);prefix.insert(prefix.end(),alpha.begin(),alpha.end());for(int q=0;q<4;q++)RHS_f<<<1,96>>>(times[q],c[q]+G,r[q]+G);check(cudaGetLastError());check(cudaDeviceSynchronize());for(int trial=1;trial<=5;trial++){std::vector<double>sentinel(97,GUARD);for(int q=0;q<4;q++)check(cudaMemcpy(r[q],sentinel.data(),97*8,cudaMemcpyHostToDevice));double dt[4],sum=0;for(int q=0;q<4;q++){check(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();RHS_f<<<1,96>>>(times[q],c[q]+G,r[q]+G);check(cudaGetLastError());check(cudaDeviceSynchronize());dt[q]=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();sum+=dt[q];}bool ok=true;double err=0;int nonfinite=0;std::vector<double>output=prefix;for(int q=0;q<4;q++){std::vector<double>v(97),echo(97);check(cudaMemcpy(v.data(),r[q],97*8,cudaMemcpyDeviceToHost));check(cudaMemcpy(echo.data(),c[q],97*8,cudaMemcpyDeviceToHost));for(int i=0;i<97;i++){ok&=echo[i]==prefix[q*97+i];if(i<G||i>=G+64)ok&=v[i]==GUARD;else {double e=fabs(v[i]-gold[q*64+i-G]);if(!std::isfinite(v[i]))nonfinite++;ok&=std::isfinite(v[i])&&e<=1e-8+1e-10*fabs(gold[q*64+i-G]);err=fmax(err,e);}}output.insert(output.end(),v.begin(),v.end());}std::string file="/tmp/lci_native_"+std::to_string(getpid())+"_"+std::to_string(trial)+".bin";std::ofstream f(file,std::ios::binary);f.write((char*)output.data(),output.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"max_abs_error\":%s,\"max_rel_error\":null,\"checked_values\":256,\"nonfinite_outputs\":%d,\"jit_excluded\":true,\"timing_region\":\"sum_four_single_kernel_host_launch_plus_device_completion\",\"component_time_seconds\":[%.12g,%.12g,%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,sum,ok?"true":"false",std::isfinite(err)?std::to_string(err).c_str():"null",nonfinite,dt[0],dt[1],dt[2],dt[3],file.c_str(),output.size(),output.size());}}
