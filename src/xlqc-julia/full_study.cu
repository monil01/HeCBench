#include <cuda_runtime.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <openssl/sha.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>
#include "study_interface.h"
#include "reference_history/cuda_rys_sp.h"
#include "reference_history/cuda_rys_dp.h"
extern "C" void xl_oracle_eri_table(const double*,int,double*);
#define XL_CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)
static XLCallback xl_callback=nullptr;
static uint64_t xl_elapsed_ns=0,xl_dispatches[4]={0,0,0,0};
struct Allocation{void*base;size_t bytes;int id;};
static std::map<void*,Allocation> xl_allocations;
static std::string xl_private;static int xl_trial=0,xl_mode=0,xl_n=0,xl_basis=0,xl_nelec=0,xl_iterations=0;static double xl_nuclear=0;
static std::vector<double> xl_pbf,xl_Q,xl_S,xl_H,xl_inputs,xl_outputs,xl_stage_D,xl_stage_gold,xl_eri_full,xl_eri_stage;
static double*xl_pbf_device=nullptr,*xl_Q_device=nullptr,*xl_D_device=nullptr;static int32_t*xl_map_device=nullptr;
static std::vector<int32_t>xl_map;static std::vector<std::string>xl_captures;
static bool xl_correct=true;static double xl_max_budget_ratio=0,xl_max_stage_abs=0,xl_max_state_residual=0,xl_final_physical_residual=0,xl_final_energy_gap=0;
static std::string js(const std::string&s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';if(c=='\n'){o+="\\n";continue;}o+=c;}return o+'\"';}
static std::string number(double x){if(!std::isfinite(x))return "null";std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();}
static std::string hash(const void*p,size_t n){unsigned char d[SHA256_DIGEST_LENGTH];SHA256((const unsigned char*)p,n,d);static const char h[]="0123456789abcdef";std::string out;for(auto c:d){out+=h[c>>4];out+=h[c&15];}return out;}
static void save(const std::string&p,const void*d,size_t n){std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("captureopen");f.write((const char*)d,n);if(!f)throw std::runtime_error("capturewrite");}
static void capture(const std::string&role,const void*d,size_t n){std::string p=xl_private+"/trial"+std::to_string(xl_trial)+"_"+role+".bin";save(p,d,n);xl_captures.push_back("{\"role\":"+js(role)+",\"path\":"+js(p)+",\"bytes\":"+std::to_string(n)+",\"sha256\":"+js(hash(d,n))+"}");}
static int pair_index(int i,int j){if(i<j)std::swap(i,j);return i*(i+1)/2+j;}
static size_t ei(int i,int j,int k,int l){return ((size_t(i)*xl_n+j)*xl_n+k)*xl_n+l;}
template<class T>static std::vector<T>read_device(T*p,size_t n){std::vector<T>x(n);XL_CHECK(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
static std::vector<double>matrix(gsl_matrix*m){std::vector<double>a(m->size1*m->size2);for(size_t i=0;i<m->size1;i++)for(size_t j=0;j<m->size2;j++)a[i*m->size2+j]=gsl_matrix_get(m,i,j);return a;}
static void append(std::vector<double>&to,const std::vector<double>&v){to.insert(to.end(),v.begin(),v.end());}
static cudaError_t xl_guarded_malloc(void**out,size_t n){void*base=nullptr;XL_CHECK(cudaMalloc(&base,n+512));XL_CHECK(cudaMemset(base,0xa5,256));XL_CHECK(cudaMemset((char*)base+256,0xcd,n));XL_CHECK(cudaMemset((char*)base+256+n,0x5a,256));*out=(char*)base+256;xl_allocations[*out]={base,n,int(xl_allocations.size())};return cudaSuccess;}
static void guarded_capture(const std::string&phase){int id=0;for(auto&p:xl_allocations){auto&a=p.second;auto bytes=read_device((unsigned char*)a.base,a.bytes+512);for(int i=0;i<256;i++)xl_correct&=bytes[i]==0xa5&&bytes[a.bytes+256+i]==0x5a;capture(phase+"_allocation"+std::to_string(a.id),bytes.data(),bytes.size());}}
static cudaError_t xl_guarded_free(void*p){auto it=xl_allocations.find(p);if(it==xl_allocations.end())throw std::runtime_error("unknownallocation");auto a=it->second;auto bytes=read_device((unsigned char*)a.base,a.bytes+512);for(int i=0;i<256;i++)xl_correct&=bytes[i]==0xa5&&bytes[a.bytes+256+i]==0x5a;XL_CHECK(cudaFree(a.base));xl_allocations.erase(it);return cudaSuccess;}
static void xl_initial(int mode,int n,int b,int ne,double en,double*p,int32_t*m,double*q,double*hp,int32_t*hm,double*hq,gsl_matrix*S,gsl_matrix*T,gsl_matrix*V,gsl_matrix*H,gsl_matrix*Si){
 xl_pbf_device=p;xl_Q_device=q;xl_map_device=m;xl_mode=mode;xl_n=n;xl_basis=b;xl_nelec=ne;xl_nuclear=en;xl_pbf=read_device(p,size_t(n)*8);xl_map=read_device(m,n);xl_Q=read_device(q,b*(b+1)/2);xl_S=matrix(S);xl_H=matrix(H);xl_correct&=n==21&&b==7&&ne==10&&memcmp(hp,xl_pbf.data(),n*8*sizeof(double))==0&&memcmp(hm,xl_map.data(),n*sizeof(int32_t))==0&&memcmp(hq,xl_Q.data(),xl_Q.size()*sizeof(double))==0;
 append(xl_inputs,std::vector<double>{double(mode),double(n),double(b),double(ne),en});append(xl_inputs,xl_pbf);for(auto x:xl_map)xl_inputs.push_back(double(x));append(xl_inputs,xl_Q);append(xl_inputs,xl_S);append(xl_inputs,matrix(T));append(xl_inputs,matrix(V));append(xl_inputs,xl_H);append(xl_inputs,matrix(Si));
 static std::vector<double>cache_pbf,cache_full,cache_rounded;if(cache_pbf!=xl_pbf){cache_pbf=xl_pbf;cache_full.resize(size_t(n)*n*n*n);xl_oracle_eri_table(xl_pbf.data(),n,cache_full.data());auto rounded=xl_pbf;for(auto&x:rounded)x=double(float(x));cache_rounded.resize(cache_full.size());xl_oracle_eri_table(rounded.data(),n,cache_rounded.data());}
 xl_eri_full=cache_full;xl_eri_stage=mode?cache_full:cache_rounded;capture("actual_initial_pbf_f64",xl_pbf.data(),xl_pbf.size()*8);capture("actual_initial_pbf_to_cbf_i32",xl_map.data(),xl_map.size()*4);capture("actual_initial_Q_raw_ERI_not_sqrt_f64",xl_Q.data(),xl_Q.size()*8);capture("independent_fullERI_actualPBF_f64",xl_eri_full.data(),xl_eri_full.size()*8);
}
static void xl_before(int iter,int op,double*D,double*out,double*host){
 xl_correct &=read_device(xl_pbf_device,xl_pbf.size())==xl_pbf&&read_device(xl_Q_device,xl_Q.size())==xl_Q&&read_device(xl_map_device,xl_map.size())==xl_map;auto d=read_device(D,xl_basis*(xl_basis+1)/2);xl_correct&=memcmp(d.data(),host,d.size()*8)==0;xl_stage_D=d;xl_D_device=D;if(iter==0&&op%2==0)append(xl_inputs,d);append(xl_outputs,d);capture("iter"+std::to_string(iter)+"_op"+std::to_string(op)+"_actual_D_f64",d.data(),d.size()*8);guarded_capture("iter"+std::to_string(iter)+"_op"+std::to_string(op)+"_before");
 xl_stage_gold.assign(xl_n*(xl_n+1)/2,0);for(int i=0;i<xl_n;i++)for(int j=0;j<=i;j++){
  long double sum=0;int a=xl_map[i],b=xl_map[j];if(op%2==0){int ab=pair_index(a,b);for(int k=0;k<xl_n;k++)for(int l=0;l<=k;l++){int cd=pair_index(xl_map[k],xl_map[l]);if(std::abs(xl_Q[ab]*xl_Q[cd]*d[cd])<1e-16)continue;sum+=(long double)xl_eri_stage[ei(i,j,k,l)]*d[cd]*(k==l?1:2);}}
  else{for(int k=0;k<xl_n;k++)for(int l=0;l<xl_n;l++){int ab=pair_index(a,xl_map[k]),cd=pair_index(b,xl_map[l]),kl=pair_index(xl_map[k],xl_map[l]);if(std::abs(xl_Q[ab]*xl_Q[cd]*d[kl])<1e-16)continue;sum+=(long double)xl_eri_stage[ei(i,k,j,l)]*d[kl];}}
  xl_stage_gold[pair_index(i,j)]=double(sum);
 }
}
static void xl_dispatch(int op,int n,double*p,int32_t*m,double*d,double*out,double*q){
 XLCall call{op,n,p,m,d,out,q};if(xl_callback){if(xl_callback(&call)!=0)throw std::runtime_error("frameworkcallbackrejected");}
 else{dim3 grid(n,n),block(8,8);if(op==0)cuda_mat_J_PI<<<grid,block>>>(p,m,n,d,out,q);else if(op==1)cuda_mat_K_PI<<<grid,block>>>(p,m,n,d,out,q);else if(op==2)cuda_mat_J_PI_dp<<<grid,block>>>(p,m,n,d,out,q);else if(op==3)cuda_mat_K_PI_dp<<<grid,block>>>(p,m,n,d,out,q);else throw std::runtime_error("badopcode");}
 xl_dispatches[op]++;
}
static void xl_after(int iter,int op,double*out,double*host){
 xl_correct&=read_device(xl_pbf_device,xl_pbf.size())==xl_pbf&&read_device(xl_Q_device,xl_Q.size())==xl_Q&&read_device(xl_map_device,xl_map.size())==xl_map&&read_device(xl_D_device,xl_stage_D.size())==xl_stage_D;XL_CHECK(cudaGetLastError());auto v=read_device(out,xl_n*(xl_n+1)/2);xl_correct&=memcmp(v.data(),host,v.size()*8)==0;double absbudget=xl_mode?5e-10:1e-5,relbudget=xl_mode?5e-9:3e-5;
 for(size_t i=0;i<v.size();i++){double error=std::abs(v[i]-xl_stage_gold[i]),budget=absbudget+relbudget*std::abs(xl_stage_gold[i]);double ratio=error/budget;xl_max_stage_abs=std::max(xl_max_stage_abs,error);xl_max_budget_ratio=std::max(xl_max_budget_ratio,ratio);xl_correct&=std::isfinite(v[i])&&ratio<=1;}
 append(xl_outputs,v);capture("iter"+std::to_string(iter)+"_op"+std::to_string(op)+"_fulloutput_f64",v.data(),v.size()*8);capture("iter"+std::to_string(iter)+"_op"+std::to_string(op)+"_independent_expected_f64",xl_stage_gold.data(),xl_stage_gold.size()*8);guarded_capture("iter"+std::to_string(iter)+"_op"+std::to_string(op)+"_after");
}
static void xl_iteration(int iter,double energy,double delta,double rms,double dd,gsl_matrix*D,gsl_matrix*F,gsl_matrix*C,gsl_vector*ev,gsl_matrix*J,gsl_matrix*K){
 auto d=matrix(D),f=matrix(F),coef=matrix(C),j=matrix(J),k=matrix(K);std::vector<double>e(xl_basis);for(int i=0;i<xl_basis;i++)e[i]=gsl_vector_get(ev,i);xl_iterations=iter+1;
 append(xl_outputs,std::vector<double>{double(iter),energy,delta,rms,dd});append(xl_outputs,d);append(xl_outputs,f);append(xl_outputs,j);append(xl_outputs,k);append(xl_outputs,e);
 capture("iter"+std::to_string(iter)+"_complete_C_raw_phase_f64",coef.data(),coef.size()*8);
 double trace=0,maxorth=0,maxidem=0,maxeig=0;int n=xl_basis;
 for(int a=0;a<n;a++)for(int b=0;b<n;b++){trace+=d[a*n+b]*xl_S[b*n+a];double csc=0,dsd=0,eigen=0;for(int u=0;u<n;u++){eigen+=(f[a*n+u]-e[b]*xl_S[a*n+u])*coef[u*n+b];for(int v=0;v<n;v++){csc+=coef[u*n+a]*xl_S[u*n+v]*coef[v*n+b];dsd+=d[a*n+u]*xl_S[u*n+v]*d[v*n+b];}}maxorth=std::max(maxorth,std::abs(csc-double(a==b)));maxidem=std::max(maxidem,std::abs(dsd-2*d[a*n+b]));maxeig=std::max(maxeig,std::abs(eigen));}
 xl_max_state_residual=std::max({xl_max_state_residual,maxorth,maxidem,maxeig,std::abs(trace-xl_nelec)});xl_correct&=std::isfinite(energy)&&maxorth<1e-8&&maxidem<1e-8&&maxeig<1e-8&&std::abs(trace-xl_nelec)<1e-8;
 std::fprintf(stderr,"XLQC independent iter=%d mode=%d E=%.12f stage_budgetratio=%.5g trace=%.12g orth=%.3g idem=%.3g eig=%.3g pass=%d\n",iter,xl_mode,energy,xl_max_budget_ratio,trace,maxorth,maxidem,maxeig,int(xl_correct));std::fflush(stderr);
}
static void xl_final(int iter,double energy,gsl_matrix*D,gsl_matrix*F,gsl_matrix*C,gsl_vector*ev){
 int n=xl_basis;auto d=matrix(D),f=matrix(F);std::vector<double>physical=xl_H;
 for(int i=0;i<xl_n;i++)for(int j=0;j<xl_n;j++){int a=xl_map[i],b=xl_map[j];long double v=0;for(int k=0;k<xl_n;k++)for(int l=0;l<xl_n;l++){int c=xl_map[k],dd=xl_map[l];v+=(long double)d[c*n+dd]*(xl_eri_full[ei(i,j,k,l)]-.5*xl_eri_full[ei(i,k,j,l)]);}physical[a*n+b]+=double(v);}
 long double trueenergy=xl_nuclear;double residual=0;for(int a=0;a<n;a++)for(int b=0;b<n;b++){trueenergy+=.5L*d[a*n+b]*(xl_H[a*n+b]+physical[a*n+b]);long double r=0;for(int u=0;u<n;u++)for(int v=0;v<n;v++)r+=(long double)physical[a*n+u]*d[u*n+v]*xl_S[v*n+b]-(long double)xl_S[a*n+u]*d[u*n+v]*physical[v*n+b];residual=std::max(residual,double(std::abs(r)));}
 xl_final_physical_residual=residual;xl_final_energy_gap=std::abs(double(trueenergy)-energy);double budget=xl_mode?5e-8:5e-5;xl_correct&=residual<=budget&&xl_final_energy_gap<=budget&&xl_iterations==9;capture("final_fresh_unscreened_physical_F_of_newD_f64",physical.data(),physical.size()*8);
 std::fprintf(stderr,"XLQC final physical E=%.12f sourceE=%.12f gap=%.3g residual=%.3g budget=%.3g pass=%d\n",double(trueenergy),energy,xl_final_energy_gap,residual,budget,int(xl_correct));std::fflush(stderr);
}
#include "instrumented_main.cuh"
static std::vector<double>read_words(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("canonicalinputmissing");auto n=f.tellg();if(n<0||n%8)throw std::runtime_error("canonicalinputsize");std::vector<double>v(size_t(n)/8);f.seekg(0);f.read((char*)v.data(),n);if(!f)throw std::runtime_error("shortcanonicalinput");return v;}
static int study(XLCallback cb,int mode,const char*fixtures,const char*private_root,int prepare){
 xl_callback=cb;std::filesystem::create_directories(private_root);std::string pattern=std::string(private_root)+"/process_"+std::to_string(getpid())+"_XXXXXX";std::vector<char>tmp(pattern.begin(),pattern.end());tmp.push_back(0);char*created=mkdtemp(tmp.data());if(!created)throw std::runtime_error("mkdtemp");xl_private=created;std::string output=std::getenv("STUDY_OUTPUT_DIR")?std::getenv("STUDY_OUTPUT_DIR"):std::filesystem::current_path().string();std::filesystem::current_path(fixtures);std::string canonical=std::string(fixtures)+(mode?"/canonical_DP.f64.bin":"/canonical_SP.f64.bin");std::vector<double>expected;if(prepare&&std::filesystem::exists(canonical))throw std::runtime_error("refuseexistingcanonicalpreparation");if(!prepare)expected=read_words(canonical);
 std::fprintf(stderr,"STUDY_PRIVATE_RUN %s\n",xl_private.c_str());std::fflush(stderr);
 for(xl_trial=0;xl_trial<(prepare?1:6);xl_trial++){
  xl_elapsed_ns=0;xl_iterations=0;xl_correct=true;xl_inputs.clear();xl_outputs.clear();xl_captures.clear();xl_max_stage_abs=xl_max_budget_ratio=xl_max_state_residual=xl_final_physical_residual=xl_final_energy_gap=0;
  char program[]="xlqc",sp[]="sp",dp[]="dp";char*argv[]={program,mode?dp:sp,nullptr};int rc=xl_original_main(2,argv);xl_correct&=rc==0&&xl_allocations.empty();if(!prepare)xl_correct&=xl_inputs.size()==expected.size()&&memcmp(xl_inputs.data(),expected.data(),xl_inputs.size()*8)==0;
  std::vector<double>all=xl_inputs;append(all,xl_outputs);std::string p=prepare?std::string(fixtures)+(mode?"/preparation_DP.f64.bin":"/preparation_SP.f64.bin"):xl_trial?output+"/output_trial"+std::to_string(xl_trial)+".bin":xl_private+"/warmup_fullstate.f64.bin";save(p,all.data(),all.size()*8);
  std::string captures="[";for(size_t i=0;i<xl_captures.size();i++){if(i)captures+=',';captures+=xl_captures[i];}captures+=']';std::string common="\"correct\":"+std::string(xl_correct?"true":"false")+",\"input_prefix_count\":"+std::to_string(xl_inputs.size())+",\"checked_values\":"+std::to_string(all.size())+",\"iterations\":"+std::to_string(xl_iterations)+",\"source_mode\":"+js(mode?"dp":"sp")+",\"independent_primitive_max_abs_error\":"+number(xl_max_stage_abs)+",\"independent_primitive_max_budget_ratio\":"+number(xl_max_budget_ratio)+",\"max_S_metric_state_residual\":"+number(xl_max_state_residual)+",\"final_unscreened_physical_commutator_max_abs\":"+number(xl_final_physical_residual)+",\"final_source_vs_physical_energy_gap\":"+number(xl_final_energy_gap)+",\"raw_C_identity_claim\":false,\"raw_C_full_auxiliary_captures_and_eigen_orthogonality_checks\":true,\"full_physical_captures\":"+captures;
  if(prepare){save(canonical,xl_inputs.data(),xl_inputs.size()*8);std::ofstream f(std::string(fixtures)+(mode?"/preparation_DP.json":"/preparation_SP.json"));f<<'{'+common+'}'<<'\n';return xl_correct?0:8;}
  if(xl_trial==0){std::fprintf(stderr,"FULL_SCF_WARMUP mode%d correct%d\n",mode,int(xl_correct));if(!xl_correct)throw std::runtime_error("warmupscientificgaterejected");continue;}
  std::cout<<"STUDY_TRIAL {\"trial\":"<<xl_trial<<",\"time_seconds\":"<<number(xl_elapsed_ns*1e-9)<<",\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_all_J_K_stages_full_converged_SCF_dispatch_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":18,"<<common<<",\"output_artifact\":{\"path\":"<<js(p)<<",\"dtype\":\"float64\",\"count\":"<<all.size()<<",\"byte_order\":\"little\"}}\n";std::cout.flush();if(!xl_correct)throw std::runtime_error("checkedscientificgaterejected");
 }
 std::cout<<"STUDY_AUTHORED_DEVICE_DISPATCHES ";uint64_t total=0;for(auto n:xl_dispatches)total+=n;std::cout<<total<<" per_entry=[";for(int i=0;i<4;i++){if(i)std::cout<<',';std::cout<<xl_dispatches[i];}std::cout<<"]\n";std::cout.flush();return 0;
}
extern "C" int xl_study_main(XLCallback cb,int mode,const char*f,const char*p,int prepare){try{return study(cb,mode,f,p,prepare);}catch(const std::exception&e){std::fprintf(stderr,"XLQC_STUDY_REJECT %s\n",e.what());std::fflush(stderr);return 85;}}
#ifndef XL_SHARED
int main(int argc,char**argv){if(argc!=5)return 2;return xl_study_main(nullptr,std::atoi(argv[1]),argv[2],argv[3],std::atoi(argv[4]));}
#endif
