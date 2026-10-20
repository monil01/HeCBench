#include "configured_source.cuh"
#include <filesystem>
#include <iomanip>
#include <dlfcn.h>
static void save_floats(const std::string& p,const std::vector<float>&v){std::ofstream f(p,std::ios::binary);if(!f)abort();f.write((const char*)v.data(),v.size()*4);if(!f)abort();}
static std::vector<float> load_floats(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f){fprintf(stderr,"Missing %s\n",p.c_str());abort();}size_t z=f.tellg();if(z%4)abort();std::vector<float>v(z/4);f.seekg(0);f.read((char*)v.data(),z);if(!f)abort();return v;}
extern "C" int study_main(Callback cb,const char* fixtures,int prepare){
 study_cb=cb;Dl_info info={};dladdr((void*)cublasGetVersion_v2,&info);cublasHandle_t vh;cublasCreate(&vh);int version=-1;cublasGetVersion(vh,&version);cublasDestroy(vh);printf("STUDY_VENDOR_BINDING cublas_version=%d cublas_path=%s\n",version,info.dli_fname?info.dli_fname:"unknown");const std::string F=fixtures;std::filesystem::create_directories(F);
 char name[]="rbc-study",steps[]="1000";char*argv[]={name,steps,nullptr};
 int count=prepare?1:6;std::vector<float> input_gold,output_gold;std::vector<double> oracle;
 if(!prepare){std::ifstream report(F+"/independent_oracle_report.json");std::string report_text((std::istreambuf_iterator<char>(report)),std::istreambuf_iterator<char>());if(!report||report_text.find("\"pass\":true")==std::string::npos){fprintf(stderr,"Independent CPUgate absent/rejected\n");return 7;}input_gold=load_floats(F+"/actual_inputs.bin");std::ifstream f(F+"/independent_oracle.bin",std::ios::binary|std::ios::ate);if(!f)abort();size_t z=f.tellg();if(z%8)abort();oracle.resize(z/8);f.seekg(0);f.read((char*)oracle.data(),z);if(!f)abort();}
 for(int trial=0;trial<count;trial++){
  int rc=source_main(2,argv);CUDA_CHECK(cudaGetLastError());if(rc)return rc;
  if(prepare){save_floats(F+"/actual_inputs.bin",study_inputs);save_floats(F+"/native_prepare_outputs.bin",study_outputs);std::ofstream m(F+"/prepare_sizes.json");m<<"{\"input_float32_count\":"<<study_inputs.size()<<",\"output_float32_count\":"<<study_outputs.size()<<",\"nu_samples\":"<<study_nu_history.size()<<",\"seconds\":"<<std::setprecision(17)<<study_seconds<<"}\n";return 0;}
  bool good=study_inputs.size()==input_gold.size()&&study_outputs.size()==oracle.size();
  if(study_inputs.size()==input_gold.size())good &=memcmp(study_inputs.data(),input_gold.data(),input_gold.size()*4)==0;
  double maxerr=0.,maxratio=0.;
  // The independent Float64 CPU trajectory predeclares one 0.3%-of-full-field-Linf
  // engineering budget per named field, with scale floor1. Not a global PDE error bound.
  std::ifstream budgets(F+"/oracle_budgets.bin",std::ios::binary);std::vector<double>b(oracle.size());budgets.read((char*)b.data(),b.size()*8);if(!budgets)abort();
  for(size_t i=0;i<study_outputs.size()&&i<oracle.size();i++){double e=std::abs(double(study_outputs[i])-oracle[i]);if(!std::isfinite(study_outputs[i])||!std::isfinite(oracle[i])||e>b[i])good=false;maxerr=std::max(maxerr,e);maxratio=std::max(maxratio,e/b[i]);}
  if(trial==0){fprintf(stderr,"Explicit whole1000step warmup correct=%d max_budget_ratio=%.17g\n",good,maxratio);if(!good)return 5;continue;}
  std::vector<float>v=study_inputs;v.insert(v.end(),study_outputs.begin(),study_outputs.end());std::string path="output_trial"+std::to_string(trial)+".bin";save_floats(path,v);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_independent_field_budget_ratio\":%.17g,\"timing_region\":\"full_source_1000step_RK3_cuBLAS_and_pointwise_loop_plus_common_history_capture\",\"jit_excluded\":true,\"internal_repeats\":1,\"independent_oracle\":\"Float64_full_RK3_trajectory_all_fields_dt_and_Nu\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,study_seconds,good?"true":"false",v.size(),maxerr,maxratio,path.c_str(),v.size());fflush(stdout);if(!good)return 6;
 }
 return 0;
}
#ifndef STUDY_SHARED
int main(int argc,char**argv){if(argc!=3)return 2;return study_main(nullptr,argv[1],atoi(argv[2]));}
#endif
