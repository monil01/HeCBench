#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include "study_support.h"
#define GPU
#include "../face-cuda/haar.h"
static bool face_replay=false;
static std::vector<int64_t>face_trace,face_codes;
static std::vector<int32_t>face_gpu_input;
template<class T>void face_vec(const T*p,size_t n){if(face_replay){face_trace.push_back(n);face_trace.insert(face_trace.end(),p,p+n);}}
void face_actual_gpu_input(int*p,int n){face_gpu_input.resize(n);cudaMemcpy(face_gpu_input.data(),p,n*4,cudaMemcpyDeviceToHost);}
void face_image(MyImage*i){if(face_replay){face_trace.push_back(i->width);face_trace.push_back(i->height);face_vec(i->data,i->width*i->height);}}
void face_integrals(MyIntImage*s,MyIntImage*q){face_vec(s->data,s->width*s->height);face_vec(q->data,q->width*q->height);}
void face_pointers(int**ps,int*base,int w,int h,int n){if(!face_replay)return;std::vector<int64_t>off(n*12);for(int i=0;i<n*12;i++){if(!ps[i])off[i]=-1;else{intptr_t d=(intptr_t)ps[i]-(intptr_t)base;if(d%4||d<0||d/4>=w*h)throw std::runtime_error("bad normalized host pointer");off[i]=d/4;if((intptr_t)(base+off[i])!=(intptr_t)ps[i])throw std::runtime_error("pointer roundtrip");}}face_vec(off.data(),off.size());}
void face_after_scan(const std::vector<MyRect>&rs){if(face_replay){face_vec(face_codes.data(),face_codes.size());std::vector<int>v;for(auto&a:rs){v.push_back(a.x);v.push_back(a.y);v.push_back(a.width);v.push_back(a.height);}face_vec(v.data(),v.size());}}
static std::unordered_map<void*,size_t>gpu_allocs;
static std::vector<int64_t>face_gpu_guards;
cudaError_t face_malloc(void**p,size_t n){unsigned char*base;auto e=cudaMalloc((void**)&base,n+64);if(e!=cudaSuccess)return e;cudaMemset(base,165,32);cudaMemset(base+32+n,165,32);*p=base+32;gpu_allocs[*p]=n;return e;}
cudaError_t face_free(void*p){auto n=gpu_allocs.at(p);unsigned char g[64];auto b=(unsigned char*)p;cudaMemcpy(g,b-32,32,cudaMemcpyDeviceToHost);cudaMemcpy(g+32,b+n,32,cudaMemcpyDeviceToHost);for(auto x:g)if(x!=165)throw std::runtime_error("GPU allocation guard changed");if(!face_replay)face_gpu_guards.insert(face_gpu_guards.end(),g,g+64);gpu_allocs.erase(p);return cudaFree(b-32);}
#define cudaMalloc face_malloc
#define cudaFree face_free
#include "study_haar_instrumented.cu"
#undef cudaMalloc
#undef cudaFree
void assign_case(const FCase&c){int pos=c.w*c.h;std::copy_n(c.input.data()+pos,25,stages_array);pos+=25;std::copy_n(c.input.data()+pos,2913*12,rectangles_array);pos+=2913*12;std::copy_n(c.input.data()+pos,2913*3,weights_array);pos+=2913*3;for(int*p:{tree_thresh_array,alpha1_array,alpha2_array}){std::copy_n(c.input.data()+pos,2913,p);pos+=2913;}std::copy_n(c.input.data()+pos,25,stages_thresh_array);}
std::vector<MyRect>invoke(const FCase&c){std::vector<unsigned char>im(c.w*c.h);std::copy_n(c.input.data(),im.size(),im.data());MyImage img{c.w,c.h,255,im.data(),1};myCascade ca{};ca.n_stages=25;ca.total_nodes=2913;ca.orig_window_size={24,24};return detectObjects(&img,{20,20},{0,0},&ca,1.2f,1,2913);}
void alloc_model(){stages_array=(int*)malloc(25*4);rectangles_array=(int*)malloc(2913*12*4);weights_array=(int*)malloc(2913*3*4);alpha1_array=(int*)malloc(2913*4);alpha2_array=(int*)malloc(2913*4);tree_thresh_array=(int*)malloc(2913*4);stages_thresh_array=(int*)malloc(25*4);scaled_rectangles_array=(int**)malloc(2913*12*sizeof(int*));}
int main(){try{auto cs=face_load();alloc_model();for(auto&c:cs){assign_case(c);invoke(c);}cudaDeviceSynchronize();for(int t=1;t<=5;t++){double sec=0;face_gpu_guards.clear();std::vector<int64_t>prefix,traces;for(auto&c:cs){assign_case(c);face_replay=false;cudaDeviceSynchronize();auto a=std::chrono::steady_clock::now();auto timed=invoke(c);cudaDeviceSynchronize();sec+=std::chrono::duration<double>(std::chrono::steady_clock::now()-a).count();prefix.insert(prefix.end(),c.input.begin(),c.input.end());prefix.insert(prefix.end(),face_gpu_input.begin(),face_gpu_input.end());face_replay=true;face_trace.clear();auto rr=invoke(c);if(rr.size()!=timed.size()||!std::equal(rr.begin(),rr.end(),timed.begin(),[](auto&a,auto&b){return a.x==b.x&&a.y==b.y&&a.width==b.width&&a.height==b.height;}))throw std::runtime_error("timed/replay rectangles differ");std::vector<int>rs;for(auto&r:rr){rs.insert(rs.end(),{r.x,r.y,r.width,r.height});}face_vec(rs.data(),rs.size());std::vector<unsigned char>draw(c.w*c.h);std::copy_n(c.input.data(),draw.size(),draw.data());MyImage im{c.w,c.h,255,draw.data(),1};for(auto&r:rr)drawRectangle(&im,r);face_vec(draw.data(),draw.size());traces.insert(traces.end(),face_trace.begin(),face_trace.end());face_replay=false;}prefix.insert(prefix.end(),traces.begin(),traces.end());prefix.insert(prefix.end(),face_gpu_guards.begin(),face_gpu_guards.end());std::string file="trial"+std::to_string(t)+".bin";std::ofstream f(file,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);f.close();bool ok=face_check(prefix.data(),prefix.size());printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"max_rel_error\":%s,\"timing_region\":\"allocation_inclusive_detect_pipeline_plus_symmetric_rectangle_input_auditcopy\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,sec,ok?"true":"false",prefix.size(),ok?"0":"null",ok?"0":"null",file.c_str(),prefix.size());if(!ok)return 1;}releaseTextClassifier();return 0;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
