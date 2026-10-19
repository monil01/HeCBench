#define main original_nn_main
#include "../nn-cuda/nearestNeighbor.cu"
#undef main
#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include "corrected_selection.h"
#include "study_cases.h"
template<class T>std::vector<T> load(const char*name,size_t count){std::ifstream f(name,std::ios::binary);std::vector<T>x(count);if(!f.read((char*)x.data(),count*sizeof(T))||f.peek()!=EOF){fprintf(stderr,"invalid asset %s\n",name);exit(2);}return x;}
uint32_t bits(float x){uint32_t z;memcpy(&z,&x,4);return z;}
__global__ void echo_parameters(float*out,int n,float lat,float lng){out[0]=lat;out[1]=lng;out[2]=float(n);}
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",int(e),cudaGetErrorString(e));exit(3);}}
struct Work {NNCase c;std::vector<LatLong> host,before,after;std::vector<float>reset,raw;LatLong* d;float*out,*echo;};
int main(){auto input=load<float>("input_f32.bin",INPUT_FLOATS);auto oracle=load<double>("oracle_f64.bin",OUTPUT_COUNT);auto exact=load<uint8_t>("oracle_exact_u8.bin",OUTPUT_COUNT);std::vector<Work> w;
 for(auto c:cases){Work x;x.c=c;x.host.resize(c.n+16);memcpy(x.host.data(),input.data()+c.offset,x.host.size()*sizeof(LatLong));x.before.resize(c.n+16);x.after.resize(c.n+16);x.reset.assign(c.n+32,91.25f);for(int i=0;i<c.n;i++)x.reset[i+16]=std::numeric_limits<float>::quiet_NaN();x.raw.resize(c.n+32);check(cudaMalloc(&x.d,x.host.size()*sizeof(LatLong)));check(cudaMalloc(&x.out,x.raw.size()*4));check(cudaMalloc(&x.echo,12));check(cudaMemcpy(x.d,x.host.data(),x.host.size()*sizeof(LatLong),cudaMemcpyHostToDevice));w.push_back(std::move(x));}
 auto batch=[](Work&x){for(int i=0;i<x.c.repeats;i++)nn<<<(x.c.n+63)/64,64>>>(x.c.n,x.c.lat,x.c.lng,x.d+8,x.out+16);check(cudaDeviceSynchronize());};
 for(auto&x:w){check(cudaMemcpy(x.out,x.reset.data(),x.reset.size()*4,cudaMemcpyHostToDevice));batch(x);}
 for(int trial=1;trial<=5;trial++){std::vector<double> prefix,output,component;double elapsed=0;for(auto&x:w){auto c=x.c;check(cudaMemcpy(x.out,x.reset.data(),x.reset.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(x.before.data(),x.d,x.before.size()*sizeof(LatLong),cudaMemcpyDeviceToHost));echo_parameters<<<1,1>>>(x.echo,c.n,c.lat,c.lng);float echoed[3];check(cudaMemcpy(echoed,x.echo,12,cudaMemcpyDeviceToHost));check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();batch(x);double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/c.repeats;component.push_back(seconds);elapsed+=seconds;check(cudaMemcpy(x.after.data(),x.d,x.after.size()*sizeof(LatLong),cudaMemcpyDeviceToHost));check(cudaMemcpy(x.raw.data(),x.out,x.raw.size()*4,cudaMemcpyDeviceToHost));
 prefix.insert(prefix.end(),{double(c.n),double(c.k),double(c.repeats),double(bits(c.lat)),double(bits(c.lng)),double(bits(echoed[0])),double(bits(echoed[1])),double(bits(echoed[2]))});for(auto&v:x.before){prefix.push_back(bits(v.lat));prefix.push_back(bits(v.lng));}for(auto&v:x.after){prefix.push_back(bits(v.lat));prefix.push_back(bits(v.lng));}
 std::vector<float>d(c.n);std::vector<Record>records(c.n);for(int i=0;i<c.n;i++){d[i]=x.raw[i+16];output.push_back(d[i]);snprintf(records[i].recString,REC_LENGTH,"ID%08d",i);}nn_corrected_findLowest(records,d.data(),c.n,c.k);for(auto v:d)output.push_back(v);for(auto&v:records)output.push_back(atoi(v.recString+2));for(int i=0;i<16;i++)output.push_back(bits(x.raw[i]));for(int i=c.n+16;i<c.n+32;i++)output.push_back(bits(x.raw[i]));}
 if(prefix.size()!=PREFIX_COUNT)return 4;prefix.insert(prefix.end(),output.begin(),output.end());if(prefix.size()!=OUTPUT_COUNT)return 5;double ma=0,mr=0;bool ok=true;for(size_t i=0;i<prefix.size();i++){double a=fabs(prefix[i]-oracle[i]);ma=fmax(ma,a);mr=fmax(mr,a/fmax(fabs(oracle[i]),1e-300));ok&=std::isfinite(prefix[i])&&(exact[i]?a==0:a<=1e-5+8*double(FLT_EPSILON)*fabs(oracle[i]));}
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/nn_trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum5_original_distance_kernel_batch_means_R10000\",\"jit_excluded\":true,\"internal_repeats\":10000,\"component_time_seconds\":[",trial,elapsed,ok?"true":"false",prefix.size(),ma,mr);for(size_t i=0;i<component.size();i++)printf("%s%.17g",i?",":"",component[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),prefix.size());fflush(stdout);if(!ok)return 6;}
 for(auto&x:w){check(cudaFree(x.d));check(cudaFree(x.out));check(cudaFree(x.echo));} }
