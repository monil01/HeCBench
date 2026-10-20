#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <openssl/sha.h>
#include "study_interface.h"
#define SS_CUDA_CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"SS CUDA %s\n",cudaGetErrorString(e));throw std::runtime_error("CUDAfailure");}}while(0)
using std::uint32_t;static_assert(sizeof(uint32_t)==4,"wordsize");
SSCallback study_callback=nullptr;std::array<unsigned long long,2>study_counts{};
constexpr size_t PAD=256;constexpr uint32_t SENTINEL=0xffffffffU;
struct Allocation{void*base;void*payload;size_t bytes;};static std::map<void*,Allocation>allocations;
struct Current{std::array<void*,4>ptrs;std::array<std::vector<unsigned char>,4>before,after;uint32_t n,p,N,groups,W;std::vector<uint32_t>gold,counts,canonical,permutation,raw;bool guards=true,inactive=true,bijection=true,inputs=true,source_final=true;std::string captures;double seconds=0;};static Current current;
static std::vector<uint32_t>all_inputs,all_outputs;static std::vector<std::string>records;static bool all_good=true;static int trial_id,case_id,current_repeats;static std::string fixture_root,private_dir,current_oracle;static double total_seconds;
static std::string sha_bytes(const void*p,size_t n){unsigned char hash[SHA256_DIGEST_LENGTH];SHA256((const unsigned char*)p,n,hash);std::ostringstream s;for(auto c:hash)s<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);return s.str();}
static void save_bytes(const std::string&p,const void*data,size_t n){std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("captureopen");f.write((const char*)data,n);if(!f)throw std::runtime_error("captureshortwrite");}
static void save_words(const std::string&p,const std::vector<uint32_t>&v){save_bytes(p,v.data(),v.size()*4);}
static std::vector<uint32_t>read_words(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("missingwordinput");auto size=f.tellg();if(size<0||size_t(size)%4)throw std::runtime_error("badwordsize");std::vector<uint32_t>v(size_t(size)/4);f.seekg(0);if(size)f.read((char*)v.data(),size);if(!f)throw std::runtime_error("shortwordinput");return v;}
static std::string capture(const std::string&role,const void*p,size_t bytes){std::string path=private_dir+"/trial"+std::to_string(trial_id)+"_case"+std::to_string(case_id)+"_"+role+".bin";save_bytes(path,p,bytes);std::ostringstream s;s<<"{\"role\":\""<<role<<"\",\"path\":\""<<path<<"\",\"bytes\":"<<bytes<<",\"sha256\":\""<<sha_bytes(p,bytes)<<"\"}";return s.str();}
void study_allocate(void**out,size_t n){void*base=nullptr;SS_CUDA_CHECK(cudaMalloc(&base,n+2*PAD));std::vector<unsigned char>init(n+2*PAD,0xff);std::fill(init.begin(),init.begin()+PAD,0xa5);std::fill(init.end()-PAD,init.end(),0x5a);SS_CUDA_CHECK(cudaMemcpy(base,init.data(),init.size(),cudaMemcpyHostToDevice));*out=(unsigned char*)base+PAD;if(allocations.count(*out))throw std::runtime_error("duplicateallocation");allocations[*out]={base,*out,n};}
void study_free(void*p){auto it=allocations.find(p);if(it==allocations.end())throw std::runtime_error("unknownallocation");SS_CUDA_CHECK(cudaFree(it->second.base));allocations.erase(it);}
static std::vector<unsigned char>read_allocation(void*p){const auto&a=allocations.at(p);std::vector<unsigned char>v(a.bytes+2*PAD);SS_CUDA_CHECK(cudaMemcpy(v.data(),a.base,v.size(),cudaMemcpyDeviceToHost));return v;}
static void add_capture(const std::string&s){if(!current.captures.empty())current.captures+=",";current.captures+=s;}
void study_before(void*text,uint32_t n,void*pattern,uint32_t p,void*result,void*count,uint32_t groups,uint32_t N,uint32_t W,const unsigned char*expected_text,const unsigned char*expected_pattern){
 if(allocations.size()!=4)throw std::runtime_error("allocationcount");current=Current{};current.ptrs={text,pattern,count,result};current.n=n;current.p=p;current.N=N;current.groups=groups;current.W=W;current.gold=read_words(current_oracle);study_counts.fill(0);
 all_inputs.insert(all_inputs.end(),{uint32_t(case_id),n,p,uint32_t(current_repeats),groups,N,W,uint32_t(PAD),uint32_t(SENTINEL)});
 for(int k=0;k<4;k++){auto v=read_allocation(current.ptrs[k]);current.before[k]=v;add_capture(capture("before_allocation"+std::to_string(k),v.data(),v.size()));if(k<2){for(auto c:v)all_inputs.push_back(c);}else{if(v.size()%4)throw std::runtime_error("unalignedwordbuffer");for(size_t i=0;i<v.size();i+=4){uint32_t x;memcpy(&x,v.data()+i,4);all_inputs.push_back(x);}}}
 for(int k=0;k<4;k++){const auto&v=current.before[k];for(size_t j=0;j<PAD;j++)current.guards &=v[j]==0xa5&&v[v.size()-PAD+j]==0x5a;if(k>=2)for(size_t j=PAD;j<v.size()-PAD;j++)current.inputs &=v[j]==0xff;}
 current.inputs &=memcmp(current.before[0].data()+PAD,expected_text,n)==0&&memcmp(current.before[1].data()+PAD,expected_pattern,p)==0;
 if(!current.guards||!current.inputs){std::ofstream failure(private_dir+"/initial_input_gate_failure.json");failure<<"{\"input_exact\":"<<(current.inputs?"true":"false")<<",\"guards_exact\":"<<(current.guards?"true":"false")<<",\"complete_captures\":["<<current.captures<<"]}\n";throw std::runtime_error("initialactualinputorguardgatefailed");}
 if(N!=n-p+1||groups!=(N+W-1)/W||W!=131072)throw std::runtime_error("sourcegeometry");SS_CUDA_CHECK(cudaDeviceSynchronize());
}
void study_roi_seconds(double seconds){current.seconds=seconds;total_seconds+=seconds;}
static unsigned char lower_ascii(unsigned char c){return c>='A'&&c<='Z'?unsigned(c-'A'+'a'):c;}
void study_device_outputs(){
 for(int k=0;k<4;k++){auto v=read_allocation(current.ptrs[k]);current.after[k]=v;add_capture(capture("after_allocation"+std::to_string(k),v.data(),v.size()));current.guards &=std::equal(v.begin(),v.begin()+PAD,current.before[k].begin())&&std::equal(v.end()-PAD,v.end(),current.before[k].end()-PAD);if(k<2)current.inputs &=v==current.before[k];}
 current.counts.resize(current.groups);current.raw.resize(current.N);memcpy(current.counts.data(),current.after[2].data()+PAD,current.groups*4);memcpy(current.raw.data(),current.after[3].data()+PAD,current.N*4);
 std::vector<uint32_t>expected(current.groups,0);for(auto pos:current.gold){if(pos>=current.N)throw std::runtime_error("oraclebounds");expected[pos/current.W]++;}current.bijection &=expected==current.counts;
 std::vector<std::pair<uint32_t,uint32_t>>entries;entries.reserve(current.gold.size());
 for(uint32_t g=0;g<current.groups;g++){uint32_t begin=g*current.W,cap=std::min(current.W,current.N-begin),count=current.counts[g];if(count>cap){current.bijection=false;count=cap;}
  for(uint32_t j=0;j<count;j++){uint32_t physical=begin+j,pos=current.raw[physical];if(pos<begin||pos>=begin+cap){current.bijection=false;continue;}bool match=true;for(uint32_t k=0;k<current.p;k++)match &=lower_ascii(current.after[0][PAD+pos+k])==lower_ascii(current.after[1][PAD+k]);current.bijection &=match;entries.emplace_back(pos,physical);}
  for(uint32_t j=count;j<cap;j++)current.inactive &=current.raw[begin+j]==SENTINEL;
 }
 std::sort(entries.begin(),entries.end());for(auto&e:entries){current.canonical.push_back(e.first);current.permutation.push_back(e.second);}current.bijection &=current.canonical==current.gold;for(size_t i=1;i<current.canonical.size();i++)current.bijection &=current.canonical[i]!=current.canonical[i-1];
 std::vector<uint32_t>reconstruction(current.N,SENTINEL);for(size_t i=0;i<entries.size();i++)reconstruction[current.permutation[i]]=current.canonical[i];current.bijection &=reconstruction==current.raw;add_capture(capture("complete_canonical_to_physical_slot_permutation_u32",current.permutation.data(),current.permutation.size()*4));
 all_good &=current.guards&&current.inputs&&current.inactive&&current.bijection;
 if(!all_good){std::ofstream failure(private_dir+"/trial"+std::to_string(trial_id)+"_case"+std::to_string(case_id)+"_rejected_before_source_verify.json");failure<<"{\"all_raw_captures\":["<<current.captures<<"],\"guard_pass\":"<<(current.guards?"true":"false")<<",\"inactive_pass\":"<<(current.inactive?"true":"false")<<",\"bijection_pass\":"<<(current.bijection?"true":"false")<<"}\n";throw std::runtime_error("independentrawgatefailedbeforesourcehostverify");}
}
void study_source_final(unsigned*counts,unsigned groups,unsigned*result,std::vector<unsigned>&source_cpu,int status){
 current.source_final=status==0&&groups==current.groups&&source_cpu==current.gold&&std::equal(current.counts.begin(),current.counts.end(),counts)&&std::equal(current.gold.begin(),current.gold.end(),result);
 // Independently replay complete host compaction from immutable raw GPU slots,
 // then compare even the source-unused physical host tail, without discarding it.
 std::vector<uint32_t>host=current.raw;size_t total=current.counts[0];for(uint32_t g=1;g<current.groups;g++){std::copy_n(current.raw.begin()+g*current.W,current.counts[g],host.begin()+total);total+=current.counts[g];}std::sort(host.begin(),host.begin()+total);current.source_final &=std::equal(host.begin(),host.end(),result);add_capture(capture("complete_source_host_final_result_u32",result,current.N*4));all_good &=current.source_final;
 int op=current.p==1?0:1;all_good &=study_counts[op]==unsigned(current_repeats)&&study_counts[1-op]==0;
 all_outputs.insert(all_outputs.end(),{uint32_t(case_id),groups,uint32_t(total)});all_outputs.insert(all_outputs.end(),current.counts.begin(),current.counts.end());all_outputs.insert(all_outputs.end(),result,result+total);
 std::ostringstream rec;rec<<std::setprecision(17)<<"{\"case\":"<<case_id<<",\"text_bytes\":"<<current.n<<",\"pattern_bytes\":"<<current.p<<",\"repeats\":"<<current_repeats<<",\"workgroups\":"<<current.groups<<",\"matches\":"<<total<<",\"time_seconds\":"<<current.seconds<<",\"source_final_sorted_and_independent_all_position_oracle\":"<<(current.source_final&&current.bijection?"true":"false")<<",\"complete_lossless_raw_bijection\":"<<(current.bijection?"true":"false")<<",\"all_guard_bytes_checked\":"<<(current.guards?"true":"false")<<",\"all_inactive_result_slots_initial_sentinel\":"<<(current.inactive?"true":"false")<<",\"full_input_allocations_unchanged\":"<<(current.inputs?"true":"false")<<",\"source_GPU_dispatches\":["<<study_counts[0]<<","<<study_counts[1]<<"],\"raw_output_identity_claim\":false,\"complete_physical_captures\":["<<current.captures<<"]}";records.push_back(rec.str());fprintf(stderr,"SS independent case=%d matches=%zu exact=%d guards=%d inactive=%d bijection=%d\n",case_id,total,int(current.source_final),int(current.guards),int(current.inactive),int(current.bijection));fflush(stderr);
 std::string base=private_dir+"/trial"+std::to_string(trial_id)+"_case"+std::to_string(case_id);std::ofstream meta(base+"_complete_case.json");meta<<rec.str()<<"\n";
}
#include "configured_main.cuh"
static int run_study(SSCallback cb,const char*fixtures,const char*private_root,int prepare){
 study_callback=cb;fixture_root=fixtures;std::setlocale(LC_CTYPE,"C");std::filesystem::create_directories(private_root);std::string templ=std::string(private_root)+"/process_"+std::to_string(getpid())+"_XXXXXX";std::vector<char>name(templ.begin(),templ.end());name.push_back(0);char*made=mkdtemp(name.data());if(!made)throw std::runtime_error("uniquecaptureroot");private_dir=made;fprintf(stderr,"STUDY_PRIVATE_RUN %s\n",private_dir.c_str());
 struct Case{std::string name,path,pattern,oracle;int repeats;};std::vector<Case>cases;std::ifstream file(fixture_root+"/cases.tsv");if(!file)throw std::runtime_error("missingcases");std::string line;while(std::getline(file,line)){std::vector<std::string>f;std::stringstream in(line);std::string x;while(std::getline(in,x,'\t'))f.push_back(x);if(f.size()!=5)throw std::runtime_error("caseformat");cases.push_back({f[0],f[2],f[3],f[4],std::stoi(f[1])});}if(cases.size()!=7)throw std::runtime_error("casecount");
 if(prepare&&std::filesystem::exists(fixture_root+"/native_preparation_report.json"))throw std::runtime_error("refuseexistingpreparation");std::vector<uint32_t>canonical;if(!prepare)canonical=read_words(fixture_root+"/canonical_actual_input_prefix.u32.bin");
 for(trial_id=0;trial_id<(prepare?1:6);trial_id++){all_inputs.clear();all_outputs.clear();records.clear();all_good=true;total_seconds=0;
  for(case_id=0;case_id<int(cases.size());case_id++){auto&c=cases[case_id];current_repeats=c.repeats;current_oracle=c.oracle;char arg0[]="ss-study";std::string repeat=std::to_string(c.repeats);char*argv[]={arg0,c.path.data(),c.pattern.data(),repeat.data(),nullptr};int status=source_main(4,argv);SS_CUDA_CHECK(cudaGetLastError());if(status||!allocations.empty())throw std::runtime_error("sourcecasefailed");}
  if(!prepare)all_good &=all_inputs==canonical;std::vector<uint32_t>artifact=all_inputs;artifact.insert(artifact.end(),all_outputs.begin(),all_outputs.end());std::string casejson="[";for(size_t i=0;i<records.size();i++){if(i)casejson+=",";casejson+=records[i];}casejson+="]";
  std::string path=prepare?fixture_root+"/native_preparation_fullstate.u32.bin":"output_trial"+std::to_string(trial_id)+".bin";save_words(prepare||trial_id>0?path:private_dir+"/warmup_fullstate.u32.bin",artifact);
  if(prepare){save_words(fixture_root+"/canonical_actual_input_prefix.u32.bin",all_inputs);std::ofstream report(fixture_root+"/native_preparation_report.json");report<<"{\"pass\":"<<(all_good?"true":"false")<<",\"input_prefix_count\":"<<all_inputs.size()<<",\"full_output_count\":"<<all_outputs.size()<<",\"raw_output_identity_claim\":false,\"case_records\":"<<casejson<<"}\n";return all_good?0:8;}
  if(trial_id==0){fprintf(stderr,"SS fullsourcewarmup correct=%d\n",int(all_good));if(!all_good)return 9;continue;}
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"full_source_repeat_loops_host_launch_control_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":20006,\"raw_output_identity_claim\":false,\"complete_keyed_value_bijection_checked\":true,\"cases\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial_id,total_seconds,all_good?"true":"false",artifact.size(),casejson.c_str(),path.c_str(),artifact.size());fflush(stdout);if(!all_good)return 10;
 }
 return 0;
}
extern "C" int study_main(SSCallback cb,const char*f,const char*p,int prepare){try{return run_study(cb,f,p,prepare);}catch(const std::exception&e){fprintf(stderr,"SS STUDY FAILURE %s\n",e.what());fflush(stderr);return 85;}}
#ifndef STUDY_SHARED
int main(int argc,char**argv){if(argc!=4)return 2;return study_main(nullptr,argv[1],argv[2],atoi(argv[3]));}
#endif
