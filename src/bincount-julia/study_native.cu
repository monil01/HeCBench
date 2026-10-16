#define main original_hecbench_main
#include "../bincount-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const bool shared=argc>1&&std::string(argv[1])=="shared";constexpr int n=262144,bins=768,repeats=10;std::vector<float>input(n);std::vector<int>expected(bins,0),got(bins);for(int i=0;i<n;i++){input[i]=(float)(i%1025)/1024.f;int bin=(int)(input[i]*bins);if(bin==bins)bin--;expected[bin]+=repeats;}float*di;int*out;if(cudaMalloc(&di,n*4)!=cudaSuccess||cudaMalloc(&out,bins*4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),n*4,cudaMemcpyHostToDevice);cudaMemset(out,0,bins*4);
 auto launch=[&](){if(shared)bincount<int,float,int,DeviceMemoryType::SHARED><<<1024,256,3072>>>(out,di,bins,0.f,1.f,n,bins);else bincount<int,float,int,DeviceMemoryType::GLOBAL><<<1024,256>>>(out,di,bins,0.f,1.f,n,bins);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,bins*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,bins*4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
