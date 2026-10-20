#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include "study_interface.h"
#include "q1_oracle.h"
#define main miniFE_source_main
#include "configured/src/main.cpp"
#undef main
using Clock=std::chrono::steady_clock;
MiniCallback mini_callback=nullptr;
static double excluded_seconds=0;static bool scientific_ok=true,all_readonly_unchanged=true,prepare_inputs=false;
static std::string input_dir,aux_dir;static int case_dim=0,trial_id=0,solutions_seen=0;
static std::vector<double>prefix,outputs,dot_history;
struct CaseReview{int dim,n,nz,iterations,heuristic;double source_norm,true_relative,boundary,matrix_error,rhs_error,min_u,max_u,center;};
static std::vector<CaseReview>reviews;static Q1Input current;static Q1MatrixReview matrix_review;
static const int*active_rp=nullptr,*active_ci=nullptr;static const double*active_A=nullptr,*active_b=nullptr;
static std::vector<long long>dispatches(11,0);
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>static std::vector<T>read_device(const T*p,size_t n){std::vector<T>v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
static void append(std::vector<double>&to,const std::vector<double>&v){to.insert(to.end(),v.begin(),v.end());}
static void save_raw(const std::string&path,const std::vector<double>&v){std::ofstream f(path,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(double));if(!f)throw std::runtime_error("Cannot save full actual diagnostic/output array");}
static bool canonical_input(const Q1Input&v){
 uint64_t header[4]={0x4d4645494e505554ULL,(uint64_t)v.dim,(uint64_t)v.n,(uint64_t)v.nz};
 std::vector<std::pair<const void*,size_t>>parts{{header,sizeof(header)},{v.rp.data(),v.rp.size()*4},{v.ci.data(),v.ci.size()*4},{v.A.data(),v.A.size()*8},{v.b.data(),v.b.size()*8},{v.x.data(),v.x.size()*8},{v.r.data(),v.r.size()*8},{v.p.data(),v.p.size()*8},{v.Ap.data(),v.Ap.size()*8}};
 std::string path=input_dir+"/actual_input_"+std::to_string(v.dim)+".bin";bool exists=std::filesystem::exists(path);
 if(!exists){if(!prepare_inputs)throw std::runtime_error("Missing bound actual input readback; run explicit native prepare first");std::ofstream f(path,std::ios::binary);for(auto&p:parts)f.write((const char*)p.first,p.second);if(!f)throw std::runtime_error("Cannot retain canonical actual input bytes");}
 std::ifstream f(path,std::ios::binary);std::vector<char>buf(1<<20);size_t total=0;bool equal=true;
 for(auto&p:parts){const char*want=(const char*)p.first;size_t off=0;while(off<p.second){size_t len=std::min(buf.size(),p.second-off);f.read(buf.data(),len);if((size_t)f.gcount()!=len||memcmp(buf.data(),want+off,len))equal=false;off+=len;total+=len;}}
 if(f.peek()!=std::char_traits<char>::eof())equal=false;
 if(!equal){std::ofstream bad(aux_dir+"/rejected_actual_input_dim"+std::to_string(v.dim)+"_trial"+std::to_string(trial_id)+".bin",std::ios::binary);for(auto&p:parts)bad.write((const char*)p.first,p.second);}
 printf("STUDY_ACTUAL_INPUT dim=%d n=%d nnz=%d full_readback_bytes=%zu canonical_exact=%s retained_file=%s\n",v.dim,v.n,v.nz,total,equal?"true":"false",path.c_str());return equal;
}
void mini_dispatch(const MiniLaunch&p){
 static_assert(sizeof(MiniLaunch)==80,"Launch packet ABI");if(!mini_callback)throw std::runtime_error("Unexpected callback dispatch without authored target");if(p.op<0||p.op>=11||mini_callback(&p))throw std::runtime_error("Authored target rejected; native fallback prohibited");dispatches[p.op]++;
}
void mini_observe_initial(int n,int nz,const int*rp,const int*ci,const double*A,const double*b,const double*x,const double*r,const double*p,const double*Ap){
 auto start=Clock::now();current=Q1Input{};current.dim=case_dim;current.n=n;current.nz=nz;current.rp=read_device(rp,n+1);current.ci=read_device(ci,nz);current.A=read_device(A,nz);current.b=read_device(b,n);current.x=read_device(x,n);current.r=read_device(r,n);current.p=read_device(p,n);current.Ap=read_device(Ap,n);
 matrix_review=q1_review_matrix(current);bool initialized=true;for(auto*v:{&current.x,&current.r,&current.p,&current.Ap})for(double a:*v)initialized&=a==0.;
 if(!matrix_review.pass||!initialized){std::vector<double>diag{(double)case_dim,(double)n,(double)nz};for(int z:current.rp)diag.push_back(z);for(int z:current.ci)diag.push_back(z);for(auto*v:{&current.A,&current.b,&current.x,&current.r,&current.p,&current.Ap})append(diag,*v);save_raw(aux_dir+"/rejected_initial_dim"+std::to_string(case_dim)+".bin",diag);throw std::runtime_error("Full independent Q1 input matrix/RHS/initialization gate failed before solver");}
 if(!canonical_input(current))throw std::runtime_error("Actual full CSR/input bytes differ from retained native device readback");
 prefix.insert(prefix.end(),{(double)case_dim,(double)n,(double)nz});append(prefix,current.b);append(prefix,current.x);active_rp=rp;active_ci=ci;active_A=A;active_b=b;excluded_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
}
void mini_observe_reduction(const double*d){auto start=Clock::now();auto v=read_device(d,256);for(double x:v)scientific_ok&=std::isfinite(x);dot_history.push_back(case_dim);dot_history.insert(dot_history.end(),v.begin(),v.end());excluded_seconds+=std::chrono::duration<double>(Clock::now()-start).count();}
void mini_observe_solution(int n,const double*x,const double*r,const double*p,const double*Ap,const double*b,const double*A,int iterations,double source_norm){
 auto start=Clock::now();if(n!=current.n)throw std::runtime_error("Unexpected final solution dimension");auto xx=read_device(x,n),rr=read_device(r,n),pp=read_device(p,n),aa=read_device(Ap,n);auto review=q1_review_solution(current,xx);
 auto final_rp=read_device(active_rp,n+1),final_ci=read_device(active_ci,current.nz);auto final_A=read_device(A,current.nz),final_b=read_device(b,n);
 bool readonly=memcmp(final_rp.data(),current.rp.data(),4*(n+1))==0&&memcmp(final_ci.data(),current.ci.data(),4*current.nz)==0&&memcmp(final_A.data(),current.A.data(),8*current.nz)==0&&memcmp(final_b.data(),current.b.data(),8*n)==0;
 all_readonly_unchanged&=readonly;scientific_ok&=review.pass&&readonly;for(auto*v:{&xx,&rr,&pp,&aa})for(double z:*v)scientific_ok&=std::isfinite(z);
 outputs.insert(outputs.end(),{(double)case_dim,(double)n,(double)current.nz});for(auto*v:{&xx,&rr,&pp,&aa,&review.residual})append(outputs,*v);
 reviews.push_back({case_dim,n,current.nz,iterations,-1,source_norm,review.relative_true_residual,review.max_boundary_abs,matrix_review.max_A_abs,matrix_review.max_b_abs,review.min_u,review.max_u,review.center});solutions_seen++;excluded_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
}
void mini_observe_source_heuristic(int code){if(reviews.empty())throw std::runtime_error("Source checker called before final full-state observation");reviews.back().heuristic=code;}
static void print_cases(){printf("[");bool first=true;for(auto&r:reviews){printf("%s{\"cells_per_dimension\":%d,\"nodes\":%d,\"nnz\":%d,\"iterations\":%d,\"original_heuristic_returncode\":%d,\"source_reported_residual\":%.17g,\"true_relative_residual\":%.17g,\"max_boundary_error\":%.17g,\"max_matrix_error\":%.17g,\"max_rhs_error\":%.17g,\"min_u\":%.17g,\"max_u\":%.17g,\"center\":%.17g}",first?"":",",r.dim,r.n,r.nz,r.iterations,r.heuristic,r.source_norm,r.true_relative,r.boundary,r.matrix_error,r.rhs_error,r.min_u,r.max_u,r.center);first=false;}printf("]");}
extern "C" int mini_run(const char*parent,const char*canonical_dir,const char*aux_base,MiniCallback cb,int prepare){
 try{
  mini_callback=cb;prepare_inputs=prepare;input_dir=canonical_dir;std::filesystem::create_directories(input_dir);aux_dir=std::string(aux_base)+"/process_"+std::to_string(getpid());std::filesystem::create_directories(aux_dir);bool all=true;
  for(trial_id=0;trial_id<=(prepare?0:5);trial_id++){
   prefix.clear();outputs.clear();dot_history.clear();reviews.clear();scientific_ok=true;all_readonly_unchanged=true;excluded_seconds=0;solutions_seen=0;ck(cudaDeviceSynchronize());auto start=Clock::now();
   for(int dim:{2,16,128}){case_dim=dim;std::string size=std::to_string(dim);char program[]="miniFE",nx[]="-nx",ny[]="-ny",nz[]="-nz";char*argv[]={program,nx,size.data(),ny,size.data(),nz,size.data(),nullptr};int code=miniFE_source_main(7,argv);if(code)scientific_ok=false;}
   ck(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(Clock::now()-start).count()-excluded_seconds;if(solutions_seen!=3||reviews.size()!=3)scientific_ok=false;
   std::vector<double>all_values=prefix;append(all_values,outputs);if(prefix.size()!=4303267||all_values.size()!=15061421)scientific_ok=false;std::string side=aux_dir+"/dot_history_trial"+std::to_string(trial_id)+".bin";save_raw(side,dot_history);
   if(trial_id==0){printf("STUDY_PREPARE_OR_WARMUP correct=%s prefix_values=%zu artifact_values=%zu cases=",scientific_ok?"true":"false",prefix.size(),all_values.size());print_cases();printf("\n");fflush(stdout);if(!scientific_ok){save_raw(aux_dir+"/rejected_warmup_full_state.bin",all_values);return 2;}continue;}
   std::string path="output_trial"+std::to_string(trial_id)+".bin";save_raw(path,all_values);
   printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"full_actual_CSR_input_bytes_identical\":true,\"all_readonly_CSR_and_rhs_unchanged\":%s,\"timing_region\":\"full_CPU_Q1_assembly_and_original_CG_pipeline_minus_added_observations\",\"jit_excluded\":true,\"internal_repeats\":1,\"cases\":",trial_id,seconds,scientific_ok?"true":"false",all_values.size(),all_readonly_unchanged?"true":"false");print_cases();printf(",\"dot_history_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"encoding\":\"records257values:dimension_then256actualpostreducevalues\"},\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",side.c_str(),dot_history.size(),path.c_str(),all_values.size());fflush(stdout);all&=scientific_ok;
  }
  printf("STUDY_ACTUAL_DISPATCH_COUNTS");for(int i=0;i<11;i++)printf(" op%d=%lld",i,dispatches[i]);printf("\n");return all?0:3;
 }catch(const std::exception&e){fprintf(stderr,"MINIFE_STUDY_REJECTED %s\n",e.what());return 4;}
}
#ifndef MINIFE_STUDY_SHARED
int main(int argc,char**argv){if(argc<4)return 5;return mini_run(argv[1],argv[2],argv[3],nullptr,argc>4&&std::string(argv[4])=="--prepare");}
#endif
