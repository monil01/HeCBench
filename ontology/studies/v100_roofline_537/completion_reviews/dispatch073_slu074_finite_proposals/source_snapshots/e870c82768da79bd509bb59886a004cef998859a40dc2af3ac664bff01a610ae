#include <cuda_runtime.h>
#include <unistd.h>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <openssl/evp.h>
#include <zlib.h>
#include "study_interface.h"
#include "independent_oracle.h"
#define SLU_CUDA_CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"SLU CUDA %s\n",cudaGetErrorString(e));throw std::runtime_error("CUDAfailure");}}while(0)
SCallback study_cb=nullptr;std::array<unsigned long long,6>study_counts{};
struct Input{unsigned n,nnz,levels;bool perturb;float pert;std::vector<unsigned>ptr,rows,diag,rptr,cidx,rdiag;std::vector<int>level;std::vector<float>initial;};
static size_t current_input_start=0,current_output_start=0;static Input current;static std::vector<double>all_inputs,all_outputs;static std::vector<std::string>case_records;static bool all_good=true,prepare_mode=false,save_private=true;static int trial_id,case_id;static double roi_seconds,case_seconds;static std::string fixture_root,private_dir;
template<class T>std::vector<T> device_read(void*p,size_t n){std::vector<T>v(n);SLU_CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
template<class T>void append_values(std::vector<double>&out,const std::vector<T>&v){for(auto x:v)out.push_back(double(x));}
static std::string num(double x){if(!std::isfinite(x))return"null";std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();}
static void save_values(const std::string&p,const std::vector<double>&v){std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("openoutput");f.write((const char*)v.data(),v.size()*8);if(!f)throw std::runtime_error("shortwrite");}
static std::vector<double>load_values(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("missingcanonicalinput");size_t n=f.tellg();if(n%8)throw std::runtime_error("badcanonicalsize");std::vector<double>v(n/8);f.seekg(0);f.read((char*)v.data(),n);if(!f)throw std::runtime_error("shortcanonicalread");return v;}
static std::string scratch_record(void*pointer,size_t bytes,const std::string&phase){
 const size_t CHUNK=8*1024*1024;std::vector<unsigned char>v(CHUNK),compressed(CHUNK+CHUNK/8+128);EVP_MD_CTX*ctx=EVP_MD_CTX_new();if(!ctx||EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr)!=1)throw std::runtime_error("sha256init");
 std::string path=private_dir+"/trial"+std::to_string(trial_id)+"_case"+std::to_string(case_id)+"_"+phase+".zlib";std::ofstream output;z_stream z{};
 if(save_private){output.open(path,std::ios::binary);if(!output||deflateInit(&z,1)!=Z_OK)throw std::runtime_error("workspacecaptureopen");}
 bool zero=true;
 for(size_t off=0;off<bytes;off+=CHUNK){size_t n=std::min(CHUNK,bytes-off);SLU_CUDA_CHECK(cudaMemcpy(v.data(),(char*)pointer+off,n,cudaMemcpyDeviceToHost));EVP_DigestUpdate(ctx,v.data(),n);for(size_t i=0;i<n;i++)if(v[i]){zero=false;break;}
  if(save_private){z.next_in=v.data();z.avail_in=(uInt)n;do{z.next_out=compressed.data();z.avail_out=(uInt)compressed.size();int rc=deflate(&z,Z_NO_FLUSH);if(rc!=Z_OK)throw std::runtime_error("workspacecompress");output.write((const char*)compressed.data(),compressed.size()-z.avail_out);}while(z.avail_in);}
 }
 if(save_private){int rc;do{z.next_out=compressed.data();z.avail_out=(uInt)compressed.size();rc=deflate(&z,Z_FINISH);if(rc!=Z_OK&&rc!=Z_STREAM_END)throw std::runtime_error("workspacefinish");output.write((const char*)compressed.data(),compressed.size()-z.avail_out);}while(rc!=Z_STREAM_END);deflateEnd(&z);output.close();if(!output)throw std::runtime_error("workspacewrite");}
 unsigned char digest[EVP_MAX_MD_SIZE];unsigned size=0;EVP_DigestFinal_ex(ctx,digest,&size);EVP_MD_CTX_free(ctx);std::ostringstream h;for(unsigned i=0;i<size;i++)h<<std::hex<<std::setw(2)<<std::setfill('0')<<int(digest[i]);all_good &=zero;
 std::ostringstream s;s<<"{\"role\":\"full_actual_internal_workspace_"<<phase<<"\",\"uncompressed_bytes\":"<<bytes<<",\"raw_sha256\":\""<<h.str()<<"\",\"all_actual_bytes_zero\":"<<(zero?"true":"false")<<",\"saved\":"<<(save_private?"true":"false")<<",\"encoding\":\"zlib_level1_lossless_actual_D2H_chunks\",\"path\":"<<(save_private?"\""+path+"\"":"null")<<"}";return s.str();
}
static std::string before_scratch,after_scratch;static bool immutable_ok;
void study_before(Symbolic_Matrix&A,void**p,float*tmp,bool perturb,float pert){
 current_input_start=all_inputs.size();current_output_start=all_outputs.size();current=Input{};current.n=A.n;current.nnz=A.nnz;current.levels=A.num_lev;current.perturb=perturb;current.pert=pert;size_t n=A.n,z=A.nnz;
 current.ptr=device_read<unsigned>(p[0],n+1);current.rows=device_read<unsigned>(p[1],z);current.initial=device_read<float>(p[2],z);current.diag=device_read<unsigned>(p[3],n);current.rptr=device_read<unsigned>(p[4],n+1);current.cidx=device_read<unsigned>(p[5],z);current.rdiag=device_read<unsigned>(p[6],n);current.level=device_read<int>(p[7],n);
 if(current.ptr.front()!=0||current.ptr.back()!=z||current.rptr.front()!=0||current.rptr.back()!=z)throw std::runtime_error("sparseptrbounds");
 std::vector<std::vector<unsigned>>transpose(n);std::vector<unsigned>seen(n,0);
 for(size_t col=0;col<n;col++){
  if(current.ptr[col]>=current.ptr[col+1]||current.diag[col]<current.ptr[col]||current.diag[col]>=current.ptr[col+1]||current.rows[current.diag[col]]!=col)throw std::runtime_error("diagonalcontract");
  for(unsigned j=current.ptr[col];j<current.ptr[col+1];j++){unsigned row=current.rows[j];if(row>=n||(j>current.ptr[col]&&current.rows[j-1]>=row)||!std::isfinite(current.initial[j]))throw std::runtime_error("CSCcontract");transpose[row].push_back(col);}
  int l=current.level[col];if(l<0||unsigned(l)>=n||seen[l]++)throw std::runtime_error("levelpermutation");
 }
 for(size_t r=0;r<n;r++){if(current.rptr[r+1]-current.rptr[r]!=transpose[r].size()||current.rdiag[r]<current.rptr[r]||current.rdiag[r]>=current.rptr[r+1]||current.cidx[current.rdiag[r]]!=r)throw std::runtime_error("CSRcontract");for(size_t j=0;j<transpose[r].size();j++)if(transpose[r][j]!=current.cidx[current.rptr[r]+j])throw std::runtime_error("fullCSRtranspose");}
 all_inputs.insert(all_inputs.end(),{double(case_id),double(n),double(z),double(A.num_lev),double(perturb),double(pert),10353.});append_values(all_inputs,current.ptr);append_values(all_inputs,current.rows);append_values(all_inputs,current.initial);append_values(all_inputs,current.diag);append_values(all_inputs,current.rptr);append_values(all_inputs,current.cidx);append_values(all_inputs,current.rdiag);append_values(all_inputs,current.level);
 before_scratch=scratch_record(tmp,size_t(10353)*n*4,"before");case_seconds=0;study_counts.fill(0);immutable_ok=true;
}
void study_roi_seconds(double x){case_seconds=x;roi_seconds+=x;}
void study_after(Symbolic_Matrix&A,void**p,float*tmp){
 auto check=[&](int k,const auto&expected){using T=typename std::decay_t<decltype(expected)>::value_type;auto got=device_read<T>(p[k],expected.size());immutable_ok &=std::memcmp(got.data(),expected.data(),got.size()*sizeof(T))==0;};
 check(0,current.ptr);check(1,current.rows);check(3,current.diag);check(4,current.rptr);check(5,current.cidx);check(6,current.rdiag);check(7,current.level);all_good &=immutable_ok;after_scratch=scratch_record(tmp,size_t(10353)*A.n*4,"after");
}
void study_solved(Symbolic_Matrix&A,const std::vector<float>&x,const char*path,bool perturb){
 SparseOracleResult r=independent_lu(current.ptr,current.rows,current.diag,current.initial,A.val,x,path);all_good &=r.pass;append_values(all_outputs,A.val);append_values(all_outputs,x);
 double minpivot=INFINITY;bool clamped=false;for(unsigned c=0;c<A.n;c++){double pivot=A.val[current.diag[c]];minpivot=std::min(minpivot,std::abs(pivot));if(perturb&&current.diag[c]+1<current.ptr[c+1]&&pivot==current.pert)clamped=true;}
 // Controlledpflaginputs are intendedinactiveclamp cases. Do not certify an active modifiedmatrix withoutitsoracle.
 if(perturb&&clamped)all_good=false;
 std::ostringstream record;record<<"{\"case\":"<<case_id<<",\"n\":"<<A.n<<",\"filled_nnz\":"<<A.nnz<<",\"levels\":"<<A.num_lev<<",\"perturb_flag\":"<<(perturb?"true":"false")<<",\"perturb_threshold\":"<<num(current.pert)<<",\"clamped_pivot_observed\":"<<(clamped?"true":"false")<<",\"minimum_abs_final_pivot\":"<<num(minpivot)<<",\"time_seconds\":"<<num(case_seconds)<<",\"independent_full_product_pass\":"<<(r.pass?"true":"false")<<",\"factor_componentwise_backward_error\":"<<num(r.componentwise)<<",\"factor_normwise_backward_error\":"<<num(r.normwise)<<",\"original_system_solve_backward_error\":"<<num(r.solve_backward)<<",\"independent_LU_product_terms\":"<<r.product_terms<<",\"all_immutable_device_arrays_byte_exact\":"<<(immutable_ok?"true":"false")<<",\"authored_entry_dispatch_counts\":[";for(int i=0;i<6;i++){if(i)record<<",";record<<study_counts[i];}record<<"],\"full_internal_workspace_captures\":["<<before_scratch<<","<<after_scratch<<"]}";case_records.push_back(record.str());
 if(save_private){std::string base=private_dir+"/trial"+std::to_string(trial_id)+"_case"+std::to_string(case_id);save_values(base+"_completed_case_inputs.f64",std::vector<double>(all_inputs.begin()+current_input_start,all_inputs.end()));save_values(base+"_completed_case_outputs.f64",std::vector<double>(all_outputs.begin()+current_output_start,all_outputs.end()));std::ofstream report(base+"_completed_case.json");report<<record.str()<<"\n";}
 fprintf(stderr,"SLU independent case=%d n=%u pass=%d component=%.12g norm=%.12g solve=%.12g terms=%llu\n",case_id,A.n,int(r.pass),r.componentwise,r.normwise,r.solve_backward,r.product_terms);fflush(stderr);
}
#include "integrated_numeric.cuh"
#include "integrated_main.cuh"
extern "C" int study_main(SCallback cb,const char*fixtures,const char*priv,int prepare){
 study_cb=cb;fixture_root=fixtures;prepare_mode=prepare;private_dir=std::string(priv)+"/process_"+std::to_string(getpid());std::filesystem::create_directories(private_dir);
 struct Case{std::string name,path;bool perturb;};std::vector<Case>cases;std::ifstream spec(fixture_root+"/cases.txt");std::string line;while(std::getline(spec,line)){size_t a=line.find('\t'),b=line.find('\t',a+1);if(a==std::string::npos||b==std::string::npos)throw std::runtime_error("casesformat");cases.push_back({line.substr(0,a),line.substr(b+1),line.substr(a+1,b-a-1)=="1"});}if(cases.size()!=4)throw std::runtime_error("casecount");
 std::vector<double>canonical;if(!prepare)canonical=load_values(fixture_root+"/canonical_actual_input_prefix.bin");
 for(trial_id=0;trial_id<(prepare?1:6);trial_id++){
  all_inputs.clear();all_outputs.clear();case_records.clear();all_good=true;roi_seconds=0;save_private=prepare||trial_id>0;
  for(case_id=0;case_id<int(cases.size());case_id++){
   auto&c=cases[case_id];char name[]="slu-study",flag[]="-i",pflag[]="-p";char*args[]={name,flag,c.path.data(),pflag,nullptr};int status=source_main(c.perturb?4:3,args);SLU_CUDA_CHECK(cudaGetLastError());if(status)return status;
  }
  if(!prepare){if(canonical.size()!=all_inputs.size()||memcmp(canonical.data(),all_inputs.data(),all_inputs.size()*8)!=0)all_good=false;}
  std::vector<double>artifact=all_inputs;artifact.insert(artifact.end(),all_outputs.begin(),all_outputs.end());std::string output=prepare?fixture_root+"/native_preparation_fullstate.bin":"output_trial"+std::to_string(trial_id)+".bin";
  if(prepare||trial_id>0)save_values(output,artifact);
  std::string casejson="[";for(size_t i=0;i<case_records.size();i++){if(i)casejson+=",";casejson+=case_records[i];}casejson+="]";
  if(prepare){save_values(fixture_root+"/canonical_actual_input_prefix.bin",all_inputs);std::ofstream f(fixture_root+"/native_preparation_report.json");f<<"{\"pass\":"<<(all_good?"true":"false")<<",\"input_prefix_count\":"<<all_inputs.size()<<",\"full_output_count\":"<<all_outputs.size()<<",\"case_records\":"<<casejson<<"}\n";return all_good?0:8;}
  if(trial_id==0){fprintf(stderr,"FullSLU warmup correct=%d\n",int(all_good));if(!all_good)return 9;continue;}
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"complete_source_numeric_level_loops_and_context_completion_wallclock\",\"jit_excluded\":true,\"internal_repeats\":1,\"cases\":%s,\"original_CPUpreprocess_symbolic_and_solve_execute_outside_ROI\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial_id,roi_seconds,all_good?"true":"false",artifact.size(),casejson.c_str(),output.c_str(),artifact.size());fflush(stdout);if(!all_good)return 10;
 }
 return 0;
}
#ifndef STUDY_SHARED
int main(int argc,char**argv){if(argc!=4)return 2;return study_main(nullptr,argv[1],argv[2],atoi(argv[3]));}
#endif
