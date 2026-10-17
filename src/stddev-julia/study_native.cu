#define main original_hecbench_main
#include "../stddev-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int d=256,n=4096,repeats=10;std::vector<float>input(d*n),got(d),expected(d);for(int i=0;i<d*n;i++)input[i]=(float)(i%1024)/1024.f;stddev_ref(expected.data(),input.data(),d,n,true);float*di,*out;if(cudaMalloc(&di,d*n*4)!=cudaSuccess||cudaMalloc(&out,d*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),d*n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){stddev(out,di,d,n,true);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,d*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,d*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),0.001));ok&=std::isfinite((double)got[j])&&e<=0.001;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
