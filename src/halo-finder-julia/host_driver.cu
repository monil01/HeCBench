// Owned measured native/Julia callback harness; full source RCB CPU tree remains.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <cassert>
#include "configured_native/RCBForceTree.h"
#include "configured_native/Partition.h"
#include "halo_dispatch.h"
using Clock=std::chrono::steady_clock;
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"HALO_CUDA %s\n",cudaGetErrorString(e));exit(7);}}
__global__ void echo_actual_inputs(HaloArgs a,uint32_t*out){
 int k=blockIdx.x*blockDim.x+threadIdx.x,total=7*a.count+4*a.count1;
 if(k<total){const float*p=nullptr;int i;if(k<7*a.count){int field=k/a.count;i=k%a.count;
  if(field==0)p=a.x;else if(field==1)p=a.y;else if(field==2)p=a.z;else if(field==3)p=a.mass;else if(field==4)p=a.vx;else if(field==5)p=a.vy;else p=a.vz;
 }else{int q=k-7*a.count,field=q/a.count1;i=q%a.count1;if(field==0)p=a.x1;else if(field==1)p=a.y1;else if(field==2)p=a.z1;else p=a.mass1;}out[k]=__float_as_uint(p[i]);}
}
struct Key{std::array<uint32_t,4>w;bool operator==(const Key&o)const{return w==o.w;}};
struct Hash{size_t operator()(const Key&k)const{size_t h=1469598103934665603ULL;for(auto w:k.w)h=(h^w)*1099511628211ULL;return h;}};
static std::unordered_map<Key,int,Hash>point_ids;static std::vector<int>seen;static std::vector<float>actual_final;static int current_n=0;
static std::vector<double>prefix,frames;static size_t leaves=0,checked=0;static double excluded=0,maxabs=0,maxrelative=0;
static uint32_t *echo_buffer=nullptr;static size_t echo_capacity=0;static std::vector<uint32_t>last_before;static std::vector<int>last_target;
static uint32_t bit(float x){uint32_t w;memcpy(&w,&x,4);return w;}
static float real(uint32_t x){float v;memcpy(&v,&x,4);return v;}
static void emit_u16(std::vector<double>&out,int x){assert(0<=x&&x<=65535);out.push_back(x&255);out.push_back((x>>8)&255);}
static void emit_u32(std::vector<double>&out,uint32_t x){for(int i=0;i<4;i++)out.push_back((x>>(8*i))&255);}
static std::vector<uint32_t>echo(const HaloArgs&a){
 size_t count=7*a.count+4*a.count1;
 if(count>echo_capacity){if(echo_buffer)ck(cudaFree(echo_buffer));ck(cudaMalloc(&echo_buffer,count*4));echo_capacity=count;}
 echo_actual_inputs<<<(count+255)/256,256>>>(a,echo_buffer);ck(cudaDeviceSynchronize());std::vector<uint32_t>x(count);ck(cudaMemcpy(x.data(),echo_buffer,count*4,cudaMemcpyDeviceToHost));return x;
}
static void observe(int phase,const HaloArgs*ap){
 auto start=Clock::now();const auto&a=*ap;ck(cudaDeviceSynchronize());
 for(auto&p:halo_allocations)halo_check_guards(p.first);
 // Source force buffers really remain mapped pinned host allocations, not HBM.
 for(auto p:{a.x,a.y,a.z,a.mass,a.x1,a.y1,a.z1,a.mass1,(const float*)a.vx,(const float*)a.vy,(const float*)a.vz}){
  cudaPointerAttributes at;ck(cudaPointerGetAttributes(&at,p));if(at.type!=cudaMemoryTypeHost||at.devicePointer!=at.hostPointer){fprintf(stderr,"HALO_MEMORY_MAPPING_CHANGED\n");exit(7);}
 }
 auto words=echo(a);
 if(phase==0){
  if(a.count<1||a.count1!=current_n||a.bx!=32||a.by!=4||a.gx!=std::min((a.count1+31)/32,32)||a.gy!=std::min((a.count+3)/4,256)){fprintf(stderr,"HALO_LIST_OR_GEOMETRY_CONTRACT count%d list%d expected%d\n",a.count,a.count1,current_n);exit(8);}
  emit_u32(frames,a.count);emit_u32(frames,a.count1);emit_u32(frames,a.gx);emit_u32(frames,a.gy);emit_u32(frames,a.bx);emit_u32(frames,a.by);emit_u32(frames,bit(a.cut2));emit_u32(frames,bit(a.soft2));emit_u32(frames,bit(a.fcoeff));
  last_target.clear();std::vector<int>neighbors;
  for(int mode=0;mode<2;mode++){
   int count=mode?a.count1:a.count,off=mode?7*a.count:0;
   for(int i=0;i<count;i++){
    Key key{{words[off+i],words[off+count+i],words[off+2*count+i],words[off+3*count+i]}};auto it=point_ids.find(key);if(it==point_ids.end()){fprintf(stderr,"HALO_ACTUAL_GPU_INPUT_NOT_CANONICAL\n");exit(8);}int id=it->second;emit_u16(frames,id);
    if(mode)neighbors.push_back(id);else{if(seen[id]++){fprintf(stderr,"HALO_TARGET_DUPLICATED\n");exit(8);}last_target.push_back(id);}
   }
  }
  std::sort(neighbors.begin(),neighbors.end());for(int i=0;i<current_n;i++)if(neighbors[i]!=i){fprintf(stderr,"HALO_DIRECT_LIMIT_NEIGHBOR_COVERAGE\n");exit(8);}
  for(int k=4*a.count;k<7*a.count;k++){if(words[k]!=0){fprintf(stderr,"HALO_NONZERO_INCOMING_VELOCITY\n");exit(8);}emit_u32(frames,words[k]);}
  last_before=std::move(words);
 }else{
  for(size_t k=0;k<words.size();k++)if(k<size_t(4*a.count)||k>=size_t(7*a.count))if(words[k]!=last_before[k]){fprintf(stderr,"HALO_READONLY_GPU_INPUT_MUTATED\n");exit(8);}
  for(int dim=0;dim<3;dim++)for(int i=0;i<a.count;i++){
   float value=real(words[(4+dim)*a.count+i]);if(!std::isfinite(value)){fprintf(stderr,"HALO_NONFINITE_FORCE\n");exit(8);}frames.push_back(value);actual_final[dim*current_n+last_target[i]]=value;
  }
  leaves++;
 }
 excluded+=std::chrono::duration<double>(Clock::now()-start).count();
}
template<class T>static std::vector<T>read(const std::string&p,size_t count){std::ifstream f(p,std::ios::binary);std::vector<T>x(count);f.read((char*)x.data(),count*sizeof(T));if(!f||f.peek()!=EOF){fprintf(stderr,"HALO_FIXTURE_SIZE %s\n",p.c_str());exit(6);}return x;}
static void near(double got,double want){double err=std::abs(got-want),budget=.05+2e-4*std::abs(want);if(!std::isfinite(got)||!std::isfinite(want)||err>budget){fprintf(stderr,"HALO_INDEPENDENT_FORCE_MISMATCH got%.17g want%.17g abs%.17g budget%.17g\n",got,want,err,budget);exit(8);}checked++;maxabs=std::max(maxabs,err);maxrelative=std::max(maxrelative,err/std::max(1.0,std::abs(want)));}
extern "C" int halo_run(HaloCallback target,int trial,const char*dir){
 static bool partition_ready=false;if(!partition_ready){Partition::initialize();partition_ready=true;}
 if(!halo_allocations.empty())return 1;halo_set_callbacks(target,observe);prefix.clear();frames.clear();leaves=0;checked=0;excluded=0;maxabs=0;maxrelative=0;double seconds=0;std::vector<int>case_leaves;
 for(int case_index:{0,2,3}){
  int n=case_index==0?10001:case_index==2?31:65;current_n=n;auto input=read<float>(std::string(dir)+"/case"+std::to_string(case_index)+"_input_Float32.bin",7*n);auto gold=read<double>(std::string(dir)+"/case"+std::to_string(case_index)+"_oracle_Float64.bin",3*n);
  point_ids.clear();seen.assign(n,0);actual_final.assign(3*n,0);
  std::array<float*,7>arrays;for(int k=0;k<7;k++){ck(halo_malloc_host(&arrays[k],n*4));memcpy(arrays[k],input.data()+k*n,n*4);halo_check_guards(arrays[k]);for(int i=0;i<n;i++)prefix.push_back(double(bit(arrays[k][i])));}
  for(int i=0;i<n;i++){Key key{{bit(arrays[0][i]),bit(arrays[1][i]),bit(arrays[2][i]),bit(arrays[6][i])}};if(!point_ids.emplace(key,i).second)return 2;}
  std::vector<float>phi(n,0);std::vector<ID_T>ids(n);for(int i=0;i<n;i++)ids[i]=i;std::vector<MASK_T>mask(n,0);POSVEL_T low[3]={0,0,0},high[3]={.5f,.5f,.5f};ForceLawNewton source_law_parameter;
  FGrid grid;size_t old_leaves=leaves;double old_excluded=excluded;ck(cudaDeviceSynchronize());auto begin=Clock::now();
  auto*tree=new RCBQuadrupoleForceTree(low,high,low,high,n,arrays[0],arrays[1],arrays[2],arrays[3],arrays[4],arrays[5],arrays[6],phi.data(),ids.data(),mask.data(),1.f,grid.rmax(),.1f,0.f,12,2,128,&source_law_parameter,1.f);
  ck(cudaDeviceSynchronize());delete tree;ck(cudaDeviceSynchronize());seconds+=std::chrono::duration<double>(Clock::now()-begin).count()-(excluded-old_excluded);case_leaves.push_back(leaves-old_leaves);
  for(int id:seen)if(id!=1){fprintf(stderr,"HALO_MISSING_TARGET\n");return 3;}
  double net[3]={0,0,0},torque[3]={0,0,0},scale=0,torque_scale=0;
  emit_u32(frames,n);
  for(int p=0;p<n;p++){
   int id=ids[p];if(id<0||id>=n)return 4;emit_u16(frames,id);
   for(int dim=0;dim<3;dim++){if(bit(arrays[dim][p])!=bit(input[dim*n+id])||arrays[6][p]!=1.f){fprintf(stderr,"HALO_CPU_REORDER_MAPPING\n");return 4;}float v=arrays[3+dim][p];if(bit(v)!=bit(actual_final[dim*n+id]))return 4;near(v,gold[dim*n+id]);frames.push_back(v);net[dim]+=v;scale+=std::abs(double(v));}
   double x=arrays[0][p],y=arrays[1][p],z=arrays[2][p],ax=arrays[3][p],ay=arrays[4][p],az=arrays[5][p];torque[0]+=y*az-z*ay;torque[1]+=z*ax-x*az;torque[2]+=x*ay-y*ax;torque_scale+=(std::abs(x)+std::abs(y)+std::abs(z))*(std::abs(ax)+std::abs(ay)+std::abs(az));
  }
  for(int k=0;k<3;k++)if(std::abs(net[k])>1e-5*scale||std::abs(torque[k])>1e-5*torque_scale){fprintf(stderr,"HALO_CENTRAL_FORCE_CONSERVATION\n");return 4;}
  for(auto*p:arrays)ck(halo_free_host(p));if(!halo_allocations.empty())return 5;
 }
 if(trial>0){
  std::vector<double>out=prefix;out.insert(out.end(),frames.begin(),frames.end());std::string file="output_trial"+std::to_string(trial)+".bin";std::ofstream f(file,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();if(!f)return 6;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"metric_scope\":\"FullindependentFloat64allpairssoftenedcutoffforce_maxrelusesmax1absreference\",\"jit_excluded\":true,\"internal_repeats\":1,\"input_prefix_count\":%zu,\"timing_region\":\"FullRCBtreebuild_listtraversal_pinnedworkspaceallocation_transfers_GPUforce_contextcompletion_minus_GPUinputecho_observation_intervals\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"scientific_GPU_dispatches\":%zu,\"case_leaves\":[%d,%d,%d],\"excluded_observation_seconds\":%.17g,\"source_host_pinned_UVA_mapping_verified\":true,\"dynamic_input_index_encoding_lossless\":true}\n",trial,seconds,checked,maxabs,maxrelative,prefix.size(),file.c_str(),out.size(),leaves,case_leaves[0],case_leaves[1],case_leaves[2],excluded);fflush(stdout);
 }
 if(echo_buffer)ck(cudaFree(echo_buffer));echo_buffer=nullptr;echo_capacity=0;return 0;
}
#ifndef HALO_LIBRARY
int main(int argc,char**argv){if(argc!=2)return 1;for(int trial=0;trial<=5;trial++){int rc=halo_run(nullptr,trial,argv[1]);if(rc)return rc;}return 0;}
#endif
