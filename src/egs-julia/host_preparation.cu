// Owned guarded CPU preparation/FFI adapter. GPU transport bodies are unchanged.
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
struct StudyAllocation{void* raw;void* payload;size_t bytes;std::string name;std::string encoding;bool mutable_state;std::vector<unsigned char> initial;};
static std::vector<StudyAllocation> allocations;
static void study_check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA EGSadapter %s\n",cudaGetErrorString(e));abort();}}
template<class T>cudaError_t study_malloc(T**out,size_t n){void*p=nullptr;cudaError_t e=cudaMalloc(&p,n+256);if(e!=cudaSuccess)return e;study_check(cudaMemset(p,0xa5,n+256));*out=reinterpret_cast<T*>(static_cast<char*>(p)+128);study_check(cudaMemset(*out,0,n));allocations.push_back({p,*out,n,"","",false,{}});return cudaSuccess;}
static cudaError_t study_free(void*p){for(auto&a:allocations)if(a.payload==p){auto e=cudaFree(a.raw);a.raw=nullptr;return e;}return cudaErrorInvalidDevicePointer;}
#define CUDA_EGS
#include "configured_native/EGS.h"
#define cudaMalloc study_malloc
#define cudaFree study_free
#include "configured_native/media.c"
#include "configured_native/init.cu"
#undef cudaMalloc
#undef cudaFree
#include "configured_native/kernels.cu"
static void label(void*p,const std::string&n,const char*encoding,bool mutable_state){for(auto&a:allocations)if(a.payload==p){a.name=n;a.encoding=encoding;a.mutable_state=mutable_state;return;}abort();}
extern "C" int egs_create(const char*dir){assert(allocations.empty());static std::string paths[6];const char*names[]={"EGS_phantom_32.egsphant","EGS_phantom.pegs4dat","MTGP_3217_0-8191.bin","si","pgs4form.dat","tungsten-80kVp-4mmAl.spectrum"};for(int i=0;i<6;i++)paths[i]=std::string(dir)+"/"+names[i];egsphant_file=paths[0].c_str();pegs_file=paths[1].c_str();MT_params_file=paths[2].c_str();photon_xsections=paths[3].c_str();atomic_ff_file=paths[4].c_str();spec_file=paths[5].c_str();init(1325631759);
// These 27 MT padding words are outside the101-word recurrence. Never read by source kernels.
study_check(cudaMemset(d_MT_statuses+MT_N,0,(MT_NUM_STATUS-MT_N)*sizeof(uint)));
label(d_MT_tables,"mt_tables","uint32",false);label(d_MT_params,"mt_params","uint32",false);label(d_MT_statuses,"mt_state","uint32",true);
label(d_stack.a,"stack_a","uint32",true);label(d_stack.b,"stack_b","uint32",true);label(d_stack.c,"stack_c","uint32",true);
label(d_total_step_counts,"step_counts","uint64",true);label(d_total_list_depth,"list_depth","uint64",true);label(d_total_num_inner_iterations,"inner_iterations","uint64",true);label(d_total_weights,"total_weight","float64",true);
label(h_source.xi,"source_edges","float32",false);label(h_source.wi,"source_weights","float32",false);label(h_source.bin,"source_alias","int32",false);
label(h_phantom.x_bounds,"x_bounds","float32",false);label(h_phantom.y_bounds,"y_bounds","float32",false);label(h_phantom.z_bounds,"z_bounds","float32",false);label(d_region_data,"regions","uint32",false);
label(d_ge,"ge","float32",false);label(d_gmfp,"gmfp","float32",false);label(d_gbr1,"gbr1","float32",false);label(d_gbr2,"gbr2","float32",false);label(d_cohe,"cohe","float32",false);label(d_pmax,"pmax","float32",false);label(d_rayleigh_data,"rayleigh","float32",false);label(d_i_array,"rayleigh_indices","uint32",false);
for(int c=0;c<4;c++){label(d_detector_scores_count[0][c],"score_count_"+std::to_string(c),"float32",true);label(d_detector_scores_energy[0][c],"score_energy_"+std::to_string(c),"float32",true);label(d_detector_totals_count[c],"total_count_"+std::to_string(c),"float64",true);label(d_detector_totals_energy[c],"total_energy_"+std::to_string(c),"float64",true);}
study_check(cudaDeviceSynchronize());for(auto&a:allocations){assert(!a.name.empty());a.initial.resize(a.bytes);study_check(cudaMemcpy(a.initial.data(),a.payload,a.bytes,cudaMemcpyDeviceToHost));}return allocations.size();}
extern "C" const char*egs_name(int i){return allocations.at(i).name.c_str();}
extern "C" const char*egs_encoding(int i){return allocations.at(i).encoding.c_str();}
extern "C" size_t egs_bytes(int i){return allocations.at(i).bytes;}
extern "C" void*egs_pointer(int i){return allocations.at(i).payload;}
extern "C" int egs_mutable(int i){return allocations.at(i).mutable_state;}
extern "C" void egs_copy(int i,void*out){auto&a=allocations.at(i);study_check(cudaMemcpy(out,a.payload,a.bytes,cudaMemcpyDeviceToHost));}
extern "C" void egs_reset(){for(auto&a:allocations)study_check(cudaMemcpy(a.payload,a.initial.data(),a.bytes,cudaMemcpyHostToDevice));study_check(cudaDeviceSynchronize());}
extern "C" void egs_sync(){study_check(cudaDeviceSynchronize());}
extern "C" void egs_native_sim(int phase){simulation_step_kernel<<<dim3(1,1),32>>>(phase==0,phase!=0);study_check(cudaGetLastError());}
extern "C" void egs_native_sum(){sum_detector_scores_kernel<<<8,1024>>>();study_check(cudaGetLastError());}
extern "C" int egs_guards(){for(auto&a:allocations){unsigned char left[128],right[128];study_check(cudaMemcpy(left,a.raw,128,cudaMemcpyDeviceToHost));study_check(cudaMemcpy(right,static_cast<char*>(a.payload)+a.bytes,128,cudaMemcpyDeviceToHost));for(int j=0;j<128;j++)if(left[j]!=0xa5||right[j]!=0xa5)return 0;}return 1;}
extern "C" int egs_readonly(){for(auto&a:allocations)if(!a.mutable_state){std::vector<unsigned char>v(a.bytes);study_check(cudaMemcpy(v.data(),a.payload,a.bytes,cudaMemcpyDeviceToHost));if(v!=a.initial)return 0;}return 1;}
extern "C" double egs_scalar(int i){const double v[]={double(h_source.n),h_source.source_point.x,h_source.source_point.y,h_source.source_point.z,h_source.rectangle_z,h_source.rectangle_min.x,h_source.rectangle_min.y,h_source.rectangle_max.x,h_source.rectangle_max.y,h_source.rectangle_size.x,h_source.rectangle_size.y,h_source.rectangle_area,h_detector.center.x,h_detector.center.y,h_detector.center.z,h_detector.d.x,h_detector.d.y,double(h_detector.N.x),double(h_detector.N.y),double(h_phantom.N.x),double(h_phantom.N.y),double(h_phantom.N.z)};assert(i>=0&&i<22);return v[i];}
extern "C" void egs_destroy(){free_all();allocations.clear();}
