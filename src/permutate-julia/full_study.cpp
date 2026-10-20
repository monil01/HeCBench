#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <array>
#include "reference_history/header.h"
#include "reference_history/kernel_functions.cuh"
#include "study_interface.h"
#include "independent_oracle.h"
namespace fs=std::filesystem;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static PCallback callback=nullptr;static fs::path canonical,aux;static bool preparing=false;static int trial=0,caseid=0;static bool allcorrect=true;static double total_time=0;static std::vector<uint8_t>prefix,outputs,compression_history;static std::vector<std::string>case_reports;
template<class T>static void append(std::vector<uint8_t>&v,const T&x){auto p=reinterpret_cast<const uint8_t*>(&x);v.insert(v.end(),p,p+sizeof(x));}
static void append_bytes(std::vector<uint8_t>&v,const void*p,size_t n){auto a=static_cast<const uint8_t*>(p);v.insert(v.end(),a,a+n);}
static void write_bytes(fs::path p,const void*x,size_t n){fs::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);if(!f||!f.write(static_cast<const char*>(x),n))throw std::runtime_error("write failed "+p.string());}
static std::vector<uint8_t>read_bytes(fs::path p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("missing "+p.string());size_t n=f.tellg();std::vector<uint8_t>x(n);f.seekg(0);f.read(reinterpret_cast<char*>(x.data()),n);if(!f)throw std::runtime_error("read failed");return x;}
static std::vector<Oracle>gold(3);static std::vector<bool>gold_loaded(3,false);
static void save_gold(fs::path p,const Oracle&o){std::ofstream f(p,std::ios::binary);uint64_t n=o.packed.size(),ns=o.stats.size();f.write((char*)&n,8);f.write((char*)&ns,8);f.write((char*)&o.binary,1);f.write((char*)o.counts.data(),216);f.write((char*)o.stats.data(),ns*sizeof(Stats));f.write((char*)o.packed.data(),n);if(!f)throw std::runtime_error("gold write failed");}
static Oracle load_gold(fs::path p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("missing CPUgold "+p.string());uint64_t n,ns;Oracle o;f.read((char*)&n,8);f.read((char*)&ns,8);if(n>400000000||ns!=2500)throw std::runtime_error("gold bounds");f.read((char*)&o.binary,1);f.read((char*)o.counts.data(),216);o.stats.resize(ns);f.read((char*)o.stats.data(),ns*sizeof(Stats));o.packed.resize(n);f.read((char*)o.packed.data(),n);if(!f)throw std::runtime_error("gold read");return o;}
struct Buffer{uint8_t*base=nullptr;size_t n;Buffer(size_t sz):n(sz){ck(cudaMalloc(&base,n+32));ck(cudaMemset(base,0xa5,n+32));}~Buffer(){if(base)cudaFree(base);}void*get(){return base+16;}std::vector<uint8_t>host(){std::vector<uint8_t>x(n);ck(cudaMemcpy(x.data(),get(),n,cudaMemcpyDeviceToHost));return x;}bool guards(){std::array<uint8_t,32>x;ck(cudaMemcpy(x.data(),base,16,cudaMemcpyDeviceToHost));ck(cudaMemcpy(x.data()+16,base+16+n,16,cudaMemcpyDeviceToHost));return std::all_of(x.begin(),x.end(),[](uint8_t b){return b==0xa5;});}};
void study_compression_observe(uint32_t iteration,double value){append(compression_history,iteration);append(compression_history,value);}
void study_cpu_observe(const double*statistics,const uint32_t*counts,bool decision){append(outputs,uint32_t(caseid));append_bytes(outputs,statistics,19*sizeof(double));append_bytes(outputs,counts,57*sizeof(uint32_t));append(outputs,uint8_t(decision));append(outputs,uint32_t(compression_history.size()));append_bytes(outputs,compression_history.data(),compression_history.size());compression_history.clear();}
bool gpu_permutation_testing(double*runtime,uint32_t*counts,double*results,double mean,double median,uint8_t*data,uint32_t size,uint32_t len,uint32_t N,uint32_t blocks,uint32_t threads){
 uint32_t blen=size==1?(len+7)/8:0,loops=(10000+N-1)/N;size_t ndata=size_t(len)*N;
 Buffer D(len),S(ndata),B(size==1?size_t(blen)*N:1),R(18*sizeof(double)),C(54*sizeof(uint32_t));
 ck(cudaMemcpy(D.get(),data,len,cudaMemcpyHostToDevice));ck(cudaMemcpy(R.get(),results,18*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemcpy(C.get(),counts,54*sizeof(uint32_t),cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());
 PLaunch args{size,len,N,blocks,threads,blen,mean,median,{D.get(),S.get(),B.get(),R.get(),C.get(),nullptr}};
 std::vector<uint8_t>actual;append(actual,uint32_t(caseid));append_bytes(actual,&args,40);auto d0=D.host(),r0=R.host(),c0=C.host();append_bytes(actual,d0.data(),d0.size());append_bytes(actual,r0.data(),r0.size());append_bytes(actual,c0.data(),c0.size());auto canon=canonical/("actual_input_case"+std::to_string(caseid)+".bin");
 if(preparing)write_bytes(canon,actual.data(),actual.size());else if(actual!=read_bytes(canon))throw std::runtime_error("actual input differs");append_bytes(prefix,actual.data(),actual.size());
 auto independent_baseline=independent_stats(std::vector<uint8_t>(data,data+len),size,mean,median,false);for(int k=0;k<18;++k)if(std::abs(independent_baseline[k]-results[k])>2e-8)throw std::runtime_error("CPU source baseline/independent stats disagree field"+std::to_string(k));
 fs::path gp=canonical/("independent_full_shuffle_and_stats_case"+std::to_string(caseid)+".bin");
 if(preparing){gold[caseid]=make_oracle(std::vector<uint8_t>(data,data+len),size,N,mean,median,results);gold_loaded[caseid]=true;save_gold(gp,gold[caseid]);*runtime=0;return false;}
 if(!gold_loaded[caseid]){gold[caseid]=load_gold(gp);gold_loaded[caseid]=true;}
 Oracle&o=gold[caseid];std::vector<uint32_t>history(loops*54);uint32_t completed=0;uint8_t passed=0;
 ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
 for(uint32_t i=0;i<loops;++i){
  if(callback){if(callback(&args))throw std::runtime_error("target callback failed");}
  else{
#ifndef PERMUTATE_STUDY_SHARED
   if(size==1){binary_shuffling_kernel<<<blocks,threads>>>((uint8_t*)S.get(),(uint8_t*)B.get(),(uint8_t*)D.get(),len,blen,N);binary_statistical_tests_kernel<<<blocks*4,threads>>>((uint32_t*)C.get(),(double*)R.get(),mean,median,(uint8_t*)S.get(),(uint8_t*)B.get(),size,len,blen,N,blocks);}
   else{shuffling_kernel<<<blocks,threads>>>((uint8_t*)S.get(),(uint8_t*)D.get(),len,N);statistical_tests_kernel<<<blocks*2,threads>>>((uint32_t*)C.get(),(double*)R.get(),mean,median,(uint8_t*)S.get(),size,len,N,blocks);}
#else
   throw std::runtime_error("selected native GPU fallback forbidden");
#endif
  }
  ck(cudaGetLastError());ck(cudaMemcpy(counts,C.get(),216,cudaMemcpyDeviceToHost));std::memcpy(history.data()+54*i,counts,216);++completed;passed=0;for(int k=0;k<18;++k)if(counts[3*k]+counts[3*k+1]>5&&counts[3*k+1]+counts[3*k+2]>5)++passed;if(passed==18)break;
 }
 ck(cudaDeviceSynchronize());auto end=std::chrono::steady_clock::now();*runtime=std::chrono::duration<double>(end-start).count();total_time+=*runtime;
 bool correct=true;for(uint32_t i=0;i<completed;++i)for(int k=0;k<54;++k)correct=correct&&(history[54*i+k]==o.counts[k]*(i+1));
 auto raw=S.host();auto packed=lossless_pack(raw,o.binary);raw.clear();raw.shrink_to_fit();bool exact_shuffle=packed==o.packed;correct&=exact_shuffle;
 // All final internal shuffle bits checked every trial; one complete durable sidecar per model/case.
 std::string sidecar="";if(trial==1||!correct){fs::path p=aux/("actual_full_shuffle_case"+std::to_string(caseid)+".bin");write_bytes(p,packed.data(),packed.size());sidecar=p.string();}
 if(size==1){auto binary=B.host();for(size_t i=0;i<blen;++i)for(uint32_t t=0;t<N;++t){uint8_t expected=0;for(int j=0;j<8;++j){size_t index=(8*i+j)*N+t;uint8_t bit=(o.packed[index/8]>>(index%8))&1;expected|=bit<<(7-j);}correct&=binary[i*N+t]==expected;}if(trial==1)write_bytes(aux/("actual_binary_conversion_case"+std::to_string(caseid)+".bin"),binary.data(),binary.size());}
 correct&=D.host()==d0&&R.host()==r0;correct&=D.guards()&&S.guards()&&B.guards()&&R.guards()&&C.guards();allcorrect&=correct;
 append(outputs,uint32_t(caseid));append(outputs,completed);append(outputs,uint32_t(history.size()/54));append_bytes(outputs,history.data(),completed*54*sizeof(uint32_t));
 std::ostringstream report;report<<std::setprecision(17)<<"{\"case\":"<<caseid<<",\"size\":"<<size<<",\"len\":"<<len<<",\"parallelism\":"<<N<<",\"time_seconds\":"<<*runtime<<",\"GPU_batches\":"<<completed<<",\"GPU_permutations_including_repeated_batches\":"<<completed*N<<",\"distinct_tid_streams\":"<<N<<",\"full_internal_shuffle_values_checked\":"<<ndata<<",\"full_internal_shuffle_cpu_exact\":"<<(exact_shuffle?"true":"false")<<",\"actual_internal_packed_capture\":\""<<sidecar<<"\",\"guards_checked\":true,\"all_counters_cpu_exact\":"<<(correct?"true":"false")<<"}";case_reports.push_back(report.str());return passed==18;
}
extern "C" int study_main(PCallback cb,const char*asset,const char*canon,const char*auxiliary,int prepare){try{
 callback=cb;canonical=canon;aux=fs::path(auxiliary)/("process_"+std::to_string(getpid()));preparing=prepare;fs::create_directories(canonical);fs::create_directories(aux);auto original=read_bytes(asset);if(original.size()!=1000000||!std::all_of(original.begin(),original.end(),[](uint8_t x){return x<=1;}))throw std::runtime_error("original asset contract");
 std::vector<std::vector<uint8_t>>cases{original,std::vector<uint8_t>(2048),std::vector<uint8_t>(4096)};uint32_t state=1234567;for(size_t c=1;c<cases.size();++c)for(auto&x:cases[c]){state=state*1664525u+1013904223u;x=c==1?uint8_t(state>>24):uint8_t((state>>24)&1);}
 int first=preparing?0:0,last=preparing?0:5;
 for(trial=first;trial<=last;++trial){prefix.clear();outputs.clear();compression_history.clear();case_reports.clear();total_time=0;allcorrect=true;auto fullstart=std::chrono::steady_clock::now();for(caseid=0;caseid<3;++caseid)permutation_testing(cases[caseid].data(),caseid==2?1:8,cases[caseid].size(),2500,10,250,false);
  if(preparing){std::cout<<"NATIVE_PREPARE_PASS independent complete source-behavior oracle, no IID scientific certification\n";return 0;}
  if(!allcorrect){std::vector<uint8_t>rejected=prefix;rejected.insert(rejected.end(),outputs.begin(),outputs.end());write_bytes(aux/("rejected_trial"+std::to_string(trial)+".bin"),rejected.data(),rejected.size());throw std::runtime_error("independent full source behavior gate rejected");}if(trial==0)continue;
  std::vector<uint8_t>artifact=prefix;artifact.insert(artifact.end(),outputs.begin(),outputs.end());fs::path path=fs::current_path()/("output_trial"+std::to_string(trial)+".bin");write_bytes(path,artifact.data(),artifact.size());double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-fullstart).count();
  std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<total_time<<",\"correct\":true,\"checked_values\":"<<artifact.size()<<",\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"original_GPU_shuffle_tests_counts_copy_and_earlystop_plus_common_counter_history_observer\",\"jit_excluded\":true,\"internal_repeats\":1,\"full_controller_wall_seconds_auxiliary\":"<<wall<<",\"input_prefix_count\":"<<prefix.size()<<",\"source_IID_statistical_validity_certified\":false,\"full_internal_shuffles_checked_each_trial\":true,\"internal_capture_retention\":\"full_trial1_per_model_only; other trials full checked but not archived\",\"cases\":[";for(size_t i=0;i<case_reports.size();++i)std::cout<<(i?",":"")<<case_reports[i];std::cout<<"],\"output_artifact\":{\"path\":\""<<path.string()<<"\",\"dtype\":\"uint8\",\"count\":"<<artifact.size()<<",\"byte_order\":\"little\"}}\n";std::cout.flush();
 }
 return 0;
 }catch(const std::exception&e){fprintf(stderr,"permutate independent study rejection: %s\n",e.what());return 2;}}
#ifndef PERMUTATE_STUDY_SHARED
int main(int argc,char**argv){if(argc<4)return 3;return study_main(nullptr,argv[1],argv[2],argv[3],argc>4&&!std::strcmp(argv[4],"--prepare"));}
#endif
