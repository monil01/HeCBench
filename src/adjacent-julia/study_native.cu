#define main original_hecbench_main
#include "../adjacent-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int n=1048576,repeats=10;std::vector<int>input(n),left(n),got(n),expected(n);for(int i=0;i<n;i++)input[i]=i%17;for(int i=0;i<n;i++)left[i]=(i%1024==0)?input[i]:input[i]-input[i-1];for(int i=0;i<n;i++)expected[i]=(i%1024==1023)?left[i]:left[i]-left[i+1];int*di,*out;if(cudaMalloc(&di,n*4)!=cudaSuccess||cudaMalloc(&out,n*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){BlockAdjDiffKernel<256><<<1024,256>>>(di,out,true);BlockAdjDiffKernel<256><<<1024,256>>>(out,out,false);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
