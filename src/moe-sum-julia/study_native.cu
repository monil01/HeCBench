#define main original_hecbench_main
#include "../moe-sum-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const bool vec=argc>1&&std::string(argv[1])=="vec4";constexpr int tokens=512,hidden=1024,topk=3,repeats=10,n=tokens*hidden;std::vector<float>input(n*topk),got(n),expected(n,0.f);for(int i=0;i<n*topk;i++)input[i]=(float)((17*i)%257-128)/128.f;for(int t=0;t<tokens;t++)for(int j=0;j<hidden;j++)for(int k=0;k<topk;k++)expected[t*hidden+j]+=input[t*hidden*topk+k*hidden+j];float*di,*out;if(cudaMalloc(&di,n*topk*4)!=cudaSuccess||cudaMalloc(&out,n*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),n*topk*4,cudaMemcpyHostToDevice);
 auto launch=[&](){if(vec)moe_sum_kernel_vec4<3><<<tokens,256>>>(out,di,hidden);else moe_sum_kernel<float,3><<<tokens,1024>>>(out,di,hidden);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,n*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.0001));ok&=std::isfinite((double)got[j])&&e<=0.0001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
