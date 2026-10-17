#define main original_hecbench_main
#include "study_original_kernels.cuh"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int n=131072,repeats=10;std::vector<__half>input(n),halfout(n);std::vector<float>got(n),expected(n);const float levels[8]={0,.5,1,1.5,2,3,4,6};for(int i=0;i<n;i++){int group=i/32;float scale=exp2f(group%9-4),value=(float)((17*i)%193-96)/16.f;if(i%32==31)value=(group%2)?-6.f:6.f;input[i]=__float2half(value*scale);}for(int group=0;group<n/32;group++){float maxval=0;for(int lane=0;lane<32;lane++)maxval=fmaxf(maxval,fabsf(__half2float(input[group*32+lane])));float scale=exp2f(floorf(log2f(maxval))-2.f);for(int lane=0;lane<32;lane++){float value=__half2float(input[group*32+lane])/scale,a=fabsf(value);int best=0;float distance=fabsf(a-levels[0]);for(int j=1;j<8;j++){float e=fabsf(a-levels[j]);if(e<distance||(e==distance&&j%2==0)){best=j;distance=e;}}expected[group*32+lane]=__half2float(__float2half(copysignf(levels[best],value)*scale));}}__half *di,*out;if(cudaMalloc(&di,n*2)!=cudaSuccess||cudaMalloc(&out,n*2)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),n*2,cudaMemcpyHostToDevice);
 auto launch=[&](){qdq_mxfp4_kernel<__half,FLOAT16_EXP_BITS,FLOAT16_MANTISSA_BITS,FLOAT16_EXP_BIAS,FLOAT16_VAL_TO_ADD,FLOAT16_SIGN_EXPONENT_MASK><<<n/128,128>>>(di,out);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*2);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(halfout.data(),out,n*2,cudaMemcpyDeviceToHost);for(int i=0;i<n;i++)got[i]=__half2float(halfout[i]);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
