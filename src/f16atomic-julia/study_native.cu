#include <cuda_runtime.h>
#include <cstdio>
#include "study_original_fp16_kernel.cuh"
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int nelems=32768,ntids=nelems/2,repeats=10;std::vector<__half>halves(512);std::vector<float>got(512),expected(512);for(int i=0;i<512;i++)expected[i]=(i%2==0)?0.f:(float)(ntids/256*repeats);__half*out;cudaMalloc(&out,512*sizeof(__half));cudaMemset(out,0,512*sizeof(__half));
 auto launch=[&](){f16AtomicOnGlobalMem<<<64,256>>>(out,ntids);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,512*sizeof(__half));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(halves.data(),out,512*sizeof(__half),cudaMemcpyDeviceToHost);for(int i=0;i<512;i++)got[i]=__half2float(halves[i]);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
