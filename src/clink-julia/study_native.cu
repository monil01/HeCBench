#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include "study_kernel.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<class T>std::vector<T> read(const char*p,size_t n){std::vector<T>v(n);FILE*f=fopen(p,"rb");if(!f||fread(v.data(),sizeof(T),n,f)!=n)exit(3);fclose(f);return v;}
int main(){const int q=N*SAMPLE_TEST_LEN,pc=q+146,repeat=10;auto input=read<float>("study_inputs_f32.bin",3*pc);auto expected=read<double>("study_expected_f64.bin",3*q);float *dx[3],*dw[3],*dy[3];for(int a=0;a<3;a++){CK(cudaMalloc(&dx[a],q*4));CK(cudaMalloc(&dw[a],146*4));CK(cudaMalloc(&dy[a],q*4));CK(cudaMemcpy(dx[a],input.data()+a*pc,q*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dw[a],input.data()+a*pc+q,146*4,cudaMemcpyHostToDevice));}
auto launch=[&](int a){auto w=dw[a];lstm_inference<<<N/WGS,WGS>>>(dx[a],w,w+20,w+120,w+140,w+145,dy[a]);};for(int a=0;a<3;a++)launch(a);CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){std::vector<double>times;for(int a=0;a<3;a++){CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int r=0;r<repeat;r++)launch(a);CK(cudaGetLastError());CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/repeat);}auto out=input;double err=0,rel=0;bool ok=true;for(int a=0;a<3;a++){std::vector<float>v(q);CK(cudaMemcpy(v.data(),dy[a],q*4,cudaMemcpyDeviceToHost));for(int i=0;i<q;i++){double e=fabs(v[i]-expected[a*q+i]);err=fmax(err,e);rel=fmax(rel,e/fmax(fabs(expected[a*q+i]),1e-5));ok&=std::isfinite(v[i])&&e<=1e-5+1e-5*fabs(expected[a*q+i]);}out.insert(out.end(),v.begin(),v.end());}all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/clink_trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 4;fwrite(out.data(),4,out.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"internal_repeats\":%d,\"jit_excluded\":true,\"timing_region\":\"three_LSTM_case_kernel_loop_means_sum\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1]+times[2],times[0],times[1],times[2],ok?"true":"false",3*q,err,rel,repeat,path.c_str(),out.size());}
for(int a=0;a<3;a++){CK(cudaFree(dx[a]));CK(cudaFree(dw[a]));CK(cudaFree(dy[a]));}puts(all?"PASS":"FAIL");return all?0:1;}
