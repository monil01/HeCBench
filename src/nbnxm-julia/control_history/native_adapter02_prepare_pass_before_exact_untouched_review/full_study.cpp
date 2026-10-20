#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <array>
#include <map>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include "study_interface.h"
#define main original_nbnxm_main
#include "reference_history/main.cu"
#undef main
#include "force_oracle.h"
namespace fs=std::filesystem;using NBClock=std::chrono::steady_clock;
static_assert(sizeof(Float3)==12&&sizeof(Float4)==16&&sizeof(nbnxn_cj4_t)==32&&sizeof(nbnxn_sci_t)==16&&sizeof(nbnxn_excl_t)==128,"SourcePODlayout");
static void nb_ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct NBCase{const char*name;bool original;int gridZ,calcShift,kind;};
static NBCase nb_cases[]={{"originalNoShift",true,3199,0,0},{"originalShift",true,3199,1,0},{"centralBipartite",false,2,0,1},{"shiftedBipartiteNoShift",false,2,0,2},{"shiftedBipartiteShift",false,2,1,2},{"centralSelf",false,1,1,3}};
struct NBBuffer{void*p;size_t bytes;};
struct NBData{NBLaunch q{};std::vector<NBBuffer>buffers;template<class T>T*alloc(size_t n){T*p;nb_ck(cudaMallocManaged(&p,n*sizeof(T)));buffers.push_back({p,n*sizeof(T)});return p;}~NBData(){for(auto&b:buffers)cudaFree(b.p);}void prefetch(){int dev=0;for(auto&b:buffers)nb_ck(cudaMemPrefetchAsync(b.p,b.bytes,dev,0));nb_ck(cudaDeviceSynchronize());}NBBytes snapshot(){NBBytes r;for(auto&b:buffers){r.emplace_back(b.bytes);nb_ck(cudaMemcpy(r.back().data(),b.p,b.bytes,cudaMemcpyDeviceToHost));}return r;}};
static void nb_init(NBData&d,const NBCase&c){
 auto&q=d.q;q.nAtoms=c.original?NUM_ATOMS:c.kind==3?64:160;q.nCJ=c.original?56881:c.gridZ;q.nSCI=c.original?4806:c.gridZ;q.nEX=c.original?19205:2;q.numTypes=32;q.gridZ=c.gridZ;q.repeat=c.original?10000:1;q.calcShift=c.calcShift;q.cutoffSq=1.f;q.beta=3.12341f;q.epsFac=138.935f;
 auto*xq=d.alloc<Float4>(q.nAtoms);auto*f=d.alloc<Float3>(q.nAtoms);auto*shift=d.alloc<Float3>(45);auto*fshift=d.alloc<Float3>(45);auto*cj=d.alloc<nbnxn_cj4_t>(q.nCJ);auto*sci=d.alloc<nbnxn_sci_t>(q.nSCI);auto*ex=d.alloc<nbnxn_excl_t>(q.nEX);auto*types=d.alloc<int>(q.nAtoms);auto*coeff=d.alloc<Float2>(1024);
 for(int i=0;i<q.nAtoms;i++){xq[i]=make_float4(1.f,.5f,.25f,.125f);f[i]=Float3(1.f,.5f,.25f);types[i]=i%2;}
 for(int s=0;s<45;s++){shift[s]=Float3(1.f,.5f,.25f);fshift[s]=Float3(1.f,.5f,.25f);}for(int j=0;j<q.nCJ;j++)cj[j]=get_cj4(j);for(int j=0;j<q.nSCI;j++)sci[j]=get_sci(j);for(int j=0;j<q.nEX;j++)ex[j]=get_excl(j);for(int j=0;j<1024;j++)coeff[j]=make_float2(.5f,.25f);
 if(!c.original){
  for(int s=0;s<45;s++){shift[s]=Float3(0,0,0);fshift[s]=Float3(.5f+s*.015625f,.25f,-.125f);}shift[5]=Float3(.125f,-.0625f,.03125f);
  for(int i=0;i<q.nAtoms;i++){f[i]=Float3(.125f+(i%4)*.03125f,-.25f+(i%3)*.015625f,.5f-(i%5)*.03125f);float charge=(i%2?.125f:-.125f);if(c.kind==3)xq[i]=make_float4(.75f*(i%8),2.f*(i/8),0.f,charge);else if(i<128)xq[i]=make_float4((i%8)*.015625f,((i/8)%8)*.015625f+(i/64)*.03125f,.015625f*(i%3),charge);else xq[i]=make_float4(.75f+(i%8)*.015625f,((i-128)/8)*.25f,.015625f*(i%5),charge);}
  for(int ti=0;ti<32;ti++)for(int tj=0;tj<32;tj++){int k=(ti+tj)%3;coeff[32*ti+tj]=make_float2(.5f+.125f*k,.25f+.0625f*k);}
  for(int e=0;e<q.nEX;e++)for(int lane=0;lane<32;lane++){uint32_t word=0xffffffffu;if(c.kind!=3&&(lane+e)%3==0)word&=~(uint32_t(1)<<(((lane%4)*8+(lane/4)%8)));ex[e].pair[lane]=word;}
  for(int b=0;b<c.gridZ;b++){sci[b]={b,c.kind==2?5:c_centralShiftIndex,b,b+1};for(int j=0;j<4;j++)cj[b].cj[j]=c.kind==3?j:16+j;for(int w=0;w<2;w++){cj[b].imei[w].imask=c.kind==3?1u:(w==0?0x000000ffu:0xff00ff00u);cj[b].imei[w].excl_ind=w;}}
 }
 for(size_t j=0;j<9;j++)q.pointers[j]=d.buffers[j].p;
}
static void nb_native_phase(const NBLaunch&q){for(int r=0;r<q.repeat;r++)nbnxmKernelTest<<<dim3(1,1,q.gridZ),dim3(8,8,1)>>>((const Float4*)q.pointers[0],(Float3*)q.pointers[1],(Float3*)q.pointers[2],(Float3*)q.pointers[3],(nbnxn_cj4_t*)q.pointers[4],(const nbnxn_sci_t*)q.pointers[5],(const nbnxn_excl_t*)q.pointers[6],(const int*)q.pointers[7],(const Float2*)q.pointers[8],q.numTypes,q.cutoffSq,q.beta,q.epsFac,q.calcShift!=0);}
static void nb_write(const std::string&path,const void*p,size_t bytes){std::ofstream f(path,std::ios::binary);f.write((const char*)p,bytes);if(!f)throw std::runtime_error("Evidencewritefailed:"+path);}
static std::map<std::string,NBBytes>nb_canonical_cache;
static void nb_check_input(const std::string&dir,const NBCase&c,const NBLaunch&q,const NBBytes&actual,bool prepare){
 auto it=nb_canonical_cache.find(c.name);if(it==nb_canonical_cache.end()){
  std::string path=dir+"/actual_input_"+c.name+".bin";NBBytes data;
  if(prepare&&!fs::exists(path)){std::ofstream f(path,std::ios::binary);uint64_t magic=0x4e42494e50555431,n=9;f.write((char*)&magic,8);f.write((const char*)&q,48);f.write((char*)&n,8);for(auto&b:actual){uint64_t sz=b.size();f.write((char*)&sz,8);f.write(b.data(),sz);}if(!f)throw std::runtime_error("Canonicalwritefailed");data=actual;}
  else{std::ifstream f(path,std::ios::binary);uint64_t magic=0,n=0;char header[48];f.read((char*)&magic,8);f.read(header,48);f.read((char*)&n,8);if(!f||magic!=0x4e42494e50555431||n!=9||memcmp(header,&q,48))throw std::runtime_error("Missing/badactualinputheader");for(size_t j=0;j<n;j++){uint64_t sz=0;f.read((char*)&sz,8);if(sz!=actual[j].size())throw std::runtime_error("Actualinputbuffershape");data.emplace_back(sz);f.read(data.back().data(),sz);}if(!f)throw std::runtime_error("Truncatedactualinput");}
  it=nb_canonical_cache.emplace(c.name,std::move(data)).first;
 }
 if(actual!=it->second)throw std::runtime_error("Actualinputbitsmismatch:"+std::string(c.name));
}
struct NBReport{const char*name;double seconds=0,max_abs=0,max_budget_ratio=0,newton=0;size_t pairs=0,excluded=0,cutoff=0,selfRejected=0;bool correct=true,readonly=true,defaultExact=true,original=false;int repeat=0,gridZ=0;};
static NBReport nb_case(const NBCase&c,int caseID,const std::string&canon,bool prepare,NBCallback cb,std::vector<double>&prefix,std::vector<double>&outputs,long long&launches){
 NBData d;nb_init(d,c);d.prefetch();
 // Preserve original first-phase untimed warmup (no active pairs). It is distinct
 // from the whole-study compile warmup before the five measurement trials.
 if(c.original&&!c.calcShift){auto warm=d.q;warm.repeat=1;if(cb){if(cb(&warm))throw std::runtime_error("Warmupcallbackfailure");}else nb_native_phase(warm);nb_ck(cudaDeviceSynchronize());launches++;}
 auto initial=d.snapshot();nb_check_input(canon,c,d.q,initial,prepare);NBOracle gold=nb_oracle(d.q,initial);if(!c.original){long double signal=0;for(int i=0;i<d.q.nAtoms;i++){auto f=nb_get<Float3>(initial[1],i);for(int k=0;k<3;k++)signal=std::max(signal,fabsl(gold.force[i][k]-f[k]));}if(!gold.pairs||signal<.05L)throw std::runtime_error("Nontrivialcontrolhasnosignal");if(c.kind==3&&gold.selfRejected<8)throw std::runtime_error("Selfcontrolnotexercised");}
 prefix.insert(prefix.end(),{double(caseID),double(d.q.nAtoms),double(d.q.nCJ),double(d.q.nSCI),double(d.q.nEX),double(d.q.numTypes),double(d.q.gridZ),double(d.q.repeat),double(d.q.calcShift),double(d.q.cutoffSq),double(d.q.beta),double(d.q.epsFac)});
 for(auto j:{1,3})for(size_t k=0;k<initial[j].size()/4;k++)prefix.push_back(nb_get<float>(initial[j],k));
 d.prefetch();nb_ck(cudaDeviceSynchronize());auto start=NBClock::now();if(cb){if(cb(&d.q))throw std::runtime_error("AuthoredGPUcallbackfailure");}else nb_native_phase(d.q);nb_ck(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(NBClock::now()-start).count();launches+=d.q.repeat;
 auto actual=d.snapshot();NBReport report;report.original=c.original;report.name=c.name;report.seconds=seconds;report.pairs=gold.pairs;report.excluded=gold.excludedPairs;report.cutoff=gold.cutoffPairs;report.selfRejected=gold.selfRejected;report.repeat=d.q.repeat;report.gridZ=d.q.gridZ;
 for(auto j:{0,2,4,5,6,7,8})report.readonly&=initial[j]==actual[j];report.correct&=report.readonly;
 std::array<long double,3>net{};long double magnitude=0;
 outputs.push_back(caseID);
 for(int which=0;which<2;which++){int j=which==0?1:3;size_t rows=which==0?d.q.nAtoms:45;auto&expected=which==0?gold.force:gold.shift;
  for(size_t row=0;row<rows;row++){auto got=nb_get<Float3>(actual[j],row),init=nb_get<Float3>(initial[j],row);for(int k=0;k<3;k++){long double delta=fabsl((long double)got[k]-expected[row][k]),budget=.002L+1e-4L*fabsl(expected[row][k]);report.max_abs=std::max(report.max_abs,(double)delta);report.max_budget_ratio=std::max(report.max_budget_ratio,(double)(delta/budget));report.correct&=std::isfinite(got[k])&&delta<=budget;outputs.push_back(got[k]);if(which==0){net[k]+=(long double)got[k]-init[k];magnitude+=fabsl(expected[row][k]-init[k]);}}}
  if(c.original){report.defaultExact&=initial[j]==actual[j];report.correct&=report.defaultExact;}
 }
 report.newton=(double)(std::max({fabsl(net[0]),fabsl(net[1]),fabsl(net[2])})/std::max(1.L,magnitude));report.correct&=report.newton<=5e-5;return report;
}
static void nb_print_reports(const std::vector<NBReport>&r){printf("[");for(size_t i=0;i<r.size();i++){auto&q=r[i];printf("%s{\"name\":\"%s\",\"time_seconds\":%.17g,\"repeat\":%d,\"grid_z\":%d,\"active_pair_occurrences_per_launch\":%zu,\"excluded_correction_pair_occurrences\":%zu,\"cutoff_rejected_pairs\":%zu,\"central_self_rejected_pairs\":%zu,\"max_independent_abs_error\":%.17g,\"max_independent_error_over_budget\":%.17g,\"normalized_Newton_residual\":%.17g,\"readonly_inputs_unchanged\":%s,\"default_zero_mask_output_bit_exact\":%s,\"correct\":%s}",i?",":"",q.name,q.seconds,q.repeat,q.gridZ,q.pairs,q.excluded,q.cutoff,q.selfRejected,q.max_abs,q.max_budget_ratio,q.newton,q.readonly?"true":"false",(q.original?(q.defaultExact?"true":"false"):"null"),q.correct?"true":"false");}printf("]");}
extern "C" int nb_run(const char*canon,const char*aux,NBCallback cb,int prep){try{fs::create_directories(canon);std::string diag=std::string(aux)+"/process_"+std::to_string(getpid());fs::create_directories(diag);long long launches=0;bool prepare=prep;
 for(int trial=0;trial<=(prepare?0:5);trial++){std::vector<double>prefix,outputs;std::vector<NBReport>reports;bool correct=true;double seconds=0;for(int c=0;c<6;c++){auto r=nb_case(nb_cases[c],c,canon,prepare,cb,prefix,outputs,launches);correct&=r.correct;seconds+=r.seconds;reports.push_back(r);}std::vector<double>values=prefix;values.insert(values.end(),outputs.begin(),outputs.end());
  if(trial==0){printf("STUDY_PREPARE_OR_WARMUP correct=%s prefix_values=%zu output_values=%zu cases=",correct?"true":"false",prefix.size(),values.size());nb_print_reports(reports);printf("\n");fflush(stdout);if(!correct){nb_write(diag+"/rejected_warmup_fullstate.bin",values.data(),values.size()*8);return 2;}continue;}
  std::string out="output_trial"+std::to_string(trial)+".bin";nb_write(out,values.data(),values.size()*8);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_of_six_resident_data_source_kernel_launch_loops_plus_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"cases\":",trial,seconds,correct?"true":"false",values.size());nb_print_reports(reports);printf(",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",out.c_str(),values.size());fflush(stdout);if(!correct)return 3;
 }printf("STUDY_ACTUAL_KERNEL_LAUNCHES %lld\n",launches);return 0;}catch(const std::exception&e){fprintf(stderr,"NBNXM_STUDY_REJECTED %s\n",e.what());return 4;}}
#ifndef NBNXM_STUDY_SHARED
int main(int argc,char**argv){if(argc<3)return 5;return nb_run(argv[1],argv[2],nullptr,argc>3&&std::string(argv[3])=="--prepare");}
#endif
