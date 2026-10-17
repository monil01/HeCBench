#define main original_hecbench_main
#include "study_original_kernels.cuh"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int m=512,n=256,repeats=10,count=m*n;std::vector<float>input(count),bias(n),gamma(n),beta(n),got(count),expected(count,0.f);for(int i=0;i<count;i++)input[i]=(float)((17*i)%257-128)/128.f;for(int j=0;j<n;j++){bias[j]=(float)((13*j)%31-15)/64.f;gamma[j]=(float)(j%7+1)/8.f;beta[j]=(float)(j%5-2)/32.f;}for(int i=0;i<repeats;i++)reference<float>(expected.data(),input.data(),bias.data(),gamma.data(),beta.data(),1e-6f,m,n);float *di,*db,*dg,*dt,*out;if(cudaMalloc(&di,count*4)!=cudaSuccess||cudaMalloc(&db,n*4)!=cudaSuccess||cudaMalloc(&dg,n*4)!=cudaSuccess||cudaMalloc(&dt,n*4)!=cudaSuccess||cudaMalloc(&out,count*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),count*4,cudaMemcpyHostToDevice);cudaMemcpy(db,bias.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dg,gamma.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(dt,beta.data(),n*4,cudaMemcpyHostToDevice);cudaMemset(out,0,count*4);
 auto launch=[&](){invokeAddBiasResidualLayerNorm<float,1>(out,di,db,dg,dt,1e-6f,m,n);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,count*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,count*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.0001));ok&=std::isfinite((double)got[j])&&e<=0.0001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
