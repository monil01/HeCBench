// Source kernels MIT Copyright Dong Wang; see original source notice included below.
#define main original_cli_main
#include "../tgvnn-cuda/src/main.cu"
#undef main
#include <vector>
#include <cstring>
#include <filesystem>
#include <string>
#include <fstream>
#include <algorithm>
#include <initializer_list>
#include <dlfcn.h>
#include <sys/stat.h>
static std::string output_root,oracle_root,private_root;
static int trial_id=0;
static bool collect_trace=false, measured=false, timer_running=false;
static double elapsed=0,max_error=0,max_ratio=0;
static size_t checked=0,trace_checked=0;
static std::chrono::steady_clock::time_point clock_start;
static std::vector<float> input_capture,final_capture;
static float2 *outer_l=nullptr,*outer_s=nullptr;
static constexpr float GUARD=12345.25f;
void fail(const std::string&s){fprintf(stderr,"STUDY_FAIL %s\n",s.c_str());exit(1);}
void pause_observer(){cuTry(cudaDeviceSynchronize());if(measured&&timer_running){elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock_start).count();timer_running=false;}}
void resume_observer(){if(measured){clock_start=std::chrono::steady_clock::now();timer_running=true;}}
template<class T>std::vector<T> load(const std::string&path,size_t n){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||size_t(f.tellg())!=n*sizeof(T))fail("Missing/short/oversized oracle "+path);f.seekg(0);std::vector<T>a(n);if(!f.read((char*)a.data(),n*sizeof(T)))fail("oracle read "+path);return a;}
template<class T>void save(const std::string&path,const std::vector<T>&a){std::ofstream f(path,std::ios::binary);if(!f||!f.write((char*)a.data(),a.size()*sizeof(T)))fail("output write "+path);}
std::vector<float> fetch(float2*p,size_t n){std::vector<float>a(2*n);cuTry(cudaMemcpy(a.data(),p,2*n*sizeof(float),cudaMemcpyDeviceToHost));return a;}
std::vector<float> fetch_real(float*p,size_t n){std::vector<float>a(n);cuTry(cudaMemcpy(a.data(),p,n*sizeof(float),cudaMemcpyDeviceToHost));return a;}
void check_field(const std::string&name,const std::vector<float>&a,bool trace=false){auto ref=load<double>(oracle_root+"/"+(trace?"trace_iteration3_":"")+name+".bin",a.size());double scale=0;for(auto x:ref){if(!std::isfinite(x))fail("nonfinite CPU oracle");scale=std::max(scale,std::abs(x));}double budget=1e-6+.002*scale;for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i]))fail("nonfinite device field "+name);double d=std::abs(double(a[i])-ref[i]);max_error=std::max(max_error,d);max_ratio=std::max(max_ratio,d/budget);if(d>budget)fail("independent field budget "+name+" index "+std::to_string(i)+" abs "+std::to_string(d)+" budget "+std::to_string(budget));}if(trace)trace_checked+=a.size();else checked+=a.size();}
void append(std::vector<float>&a,const std::vector<float>&b){a.insert(a.end(),b.begin(),b.end());}
void observe_inputs(float2*img,float2*mask,size_t n){pause_observer();input_capture=fetch(img,n);append(input_capture,fetch(mask,n));resume_observer();}
void observe_trace(const char*name,float2*p,size_t n){if(!collect_trace)return;pause_observer();auto a=fetch(p,n);check_field(name,a,true);save(private_root+"/warmup_iteration3_"+name+".bin",a);resume_observer();}
struct Field {const char*name;float2*ptr;size_t count;};
void observe_final(size_t n,size_t ns,std::initializer_list<Field>fields,float*ls,float2*ls2,float2*lu,float2*vt,float2*svt,int*info,size_t m,size_t nt,float ser0,float ser1){
 pause_observer();final_capture=input_capture;
 for(auto f:fields){auto a=fetch(f.ptr,f.count);check_field(f.name,a);append(final_capture,a);}
 check_field("ser_imgz",std::vector<float>{ser0});append(final_capture,std::vector<float>{ser0});check_field("ser_imgr",std::vector<float>{ser1});append(final_capture,std::vector<float>{ser1});
 auto sl=fetch_real(ls,ns);check_field("ls",sl);append(final_capture,sl);auto sl2=fetch(ls2,ns);check_field("ls2",sl2);append(final_capture,sl2);
 int v=-999;cuTry(cudaMemcpy(&v,info,sizeof(int),cudaMemcpyDeviceToHost));if(v!=0)fail("cuSOLVER convergence info "+std::to_string(v));
 for(auto p:{outer_l+n,outer_s+n}){auto a=fetch(p,17);if(!std::all_of(a.begin(),a.end(),[](float x){return x==GUARD;}))fail("outer output guard");append(final_capture,a);}
 if(trial_id>0){std::vector<float>a=fetch(lu,m*ns);append(a,fetch(vt,ns*nt));append(a,fetch(svt,ns*nt));save(private_root+"/trial"+std::to_string(trial_id)+"_raw_SVD_factors.bin",a);}
 resume_observer();
}
#include "observed_tgv.inc"
void resolution(const char*name,void*symbol){Dl_info d;if(!dladdr(symbol,&d))fail("dladdr library");printf("VENDOR_RESOLUTION %s %s\n",name,d.dli_fname);}
int main(int argc,char**argv){
 if(argc!=6)fail("usage oracle_dir image.ra mask.ra iterations private_root");private_root=std::string(argv[5])+"/pid"+std::to_string(getpid());std::filesystem::create_directories(private_root);oracle_root=argv[1];int iterations=atoi(argv[4]);if(iterations<1)fail("iterations");const char*od=getenv("STUDY_OUTPUT_DIR");if(!od)fail("missing STUDY_OUTPUT_DIR");output_root=od;mkdir(output_root.c_str(),0777);
 ra_t img,mask;ra_read(&img,argv[2]);ra_read(&mask,argv[3]);if(img.ndims!=3||mask.ndims!=3||img.eltype!=4||mask.eltype!=4||img.elbyte!=8||mask.elbyte!=8)fail("input RA type");size_t rows=img.dims[0],cols=img.dims[1],nt=img.dims[2],n=rows*cols*nt;if(rows<3||cols<3||nt<3||rows*cols<nt||img.size!=n*8||mask.size!=n*8)fail("geometry/bytes");for(int k=0;k<3;++k)if(img.dims[k]!=mask.dims[k])fail("mask shape");
 cuTry(cudaSetDevice(0));stream[0]=nullptr;stream[1]=nullptr;cuTry(cudaMalloc((void**)&outer_l,(n+17)*sizeof(float2)));cuTry(cudaMalloc((void**)&outer_s,(n+17)*sizeof(float2)));std::vector<float2>g(17,make_float2(GUARD,GUARD));cuTry(cudaMemcpy(outer_l+n,g.data(),g.size()*8,cudaMemcpyHostToDevice));cuTry(cudaMemcpy(outer_s+n,g.data(),g.size()*8,cudaMemcpyHostToDevice));
 resolution("cuFFT",(void*)&cufftExecC2C);resolution("cuBLAS",(void*)&cublasCgemm_64);resolution("cuSOLVER",(void*)&cusolverDnCgesvd);
 for(int t=0;t<=5;++t){trial_id=t;collect_trace=t==0;measured=t>0;elapsed=0;max_error=0;max_ratio=0;checked=0;trace_checked=0;cuTry(cudaDeviceSynchronize());resume_observer();observed_tgv_cs(outer_l,outer_s,(float2*)img.data,(float2*)mask.data,n,rows,cols,nt,.004f,.5f,1.f,.25f,.25f,.5f,iterations);pause_observer();measured=false;
  // Source does not destroy each overwritten FFT plan; explicit outside-ROI cleanup.
  if(cufftDestroy(fft2_plan)!=CUFFT_SUCCESS)fail("FFT cleanup");
  if(std::memcmp(input_capture.data(),img.data,n*8)||std::memcmp(input_capture.data()+2*n,mask.data,n*8))fail("actualdeviceinputbitoriginalRA");
  auto image=load<double>(oracle_root+"/input_img.bin",2*n);auto maskref=load<double>(oracle_root+"/input_mask.bin",2*n);for(size_t i=0;i<2*n;++i)if(input_capture[i]!=float(image[i])||input_capture[2*n+i]!=float(maskref[i]))fail("actual device input bytes");
  if(t==0){printf("UNTIMED_TRACE {\"actual_third_iteration\":true,\"checked_components\":%zu,\"full_trajectory_iterations\":%d}\n",trace_checked,iterations);continue;}
  std::string artifact=output_root+"/trial"+std::to_string(t)+".bin";save(artifact,final_capture);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_independent_field_budget_ratio\":%.17g,\"jit_excluded\":true,\"timing_valid\":true,\"input_contract_matched\":true,\"shared_inputs\":true,\"timed_region_matched\":true,\"timing_region\":\"source_tgv_cs_completed_pipeline_excluding_observers_and_formatting\",\"private_output_directory\":\"%s\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,elapsed,checked+4*n+68,max_error,max_ratio,private_root.c_str(),artifact.c_str(),final_capture.size());fflush(stdout);
 }
 cuTry(cudaFree(outer_l));cuTry(cudaFree(outer_s));ra_free(&img);ra_free(&mask);return 0;
}
