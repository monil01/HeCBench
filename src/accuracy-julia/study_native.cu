#define main original_hecbench_main
#include "../accuracy-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
constexpr int n=4096,d=128,repeats=10;std::vector<float>input(n*d);std::vector<int>labels(n),got(1),expected(1);for(int i=0;i<n*d;i++)input[i]=(17*i+13*(i/d))%23;for(int i=0;i<n;i++)labels[i]=(7*i)%d;expected[0]=reference(n,d,5,input.data(),labels.data());float*di;int*dl,*out;if(cudaMalloc(&di,n*d*4)!=cudaSuccess||cudaMalloc(&dl,n*4)!=cudaSuccess||cudaMalloc(&out,4)!=cudaSuccess)return 3;cudaMemcpy(di,input.data(),n*d*4,cudaMemcpyHostToDevice);cudaMemcpy(dl,labels.data(),n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){cudaMemsetAsync(out,0,4);accuracy_kernel<<<1024,256>>>(n,d,5,di,dl,out);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemset(out,0,4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(got.data(),out,4,cudaMemcpyDeviceToHost);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-300));ok&=std::isfinite((double)got[j])&&e<=0;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
