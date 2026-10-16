#define main original_hecbench_main
#include "study_original_kernels.cuh"
#include "study_cpu_reference.h"
#undef main
#include <vector>
#include <string>
int main(int argc,char**argv){
 bool vec=argc>1&&std::string(argv[1])=="vec2";int repeats=argc>2?atoi(argv[2]):10;if(repeats<1)return 2;
 constexpr int width=1024,height=128,batch=4;const size_t n=(size_t)width*height*batch;
 std::vector<__half>host(n),bias(width),expected(n),got(n);for(size_t i=0;i<n;i++)host[i]=__float2half((float)(i%1024)/1024.f);for(int y=0;y<width;y++)bias[y]=__float2half((float)(y%12-6));expected=host;
 for(int j=0;j<repeats;j++)gelu_bias_loop_cpu(expected.data(),bias.data(),batch,width,height);
 __half*src,*device_bias;if(cudaMalloc(&src,n*sizeof(__half))!=cudaSuccess||cudaMalloc(&device_bias,width*sizeof(__half))!=cudaSuccess)return 3;cudaMemcpy(src,host.data(),n*sizeof(__half),cudaMemcpyHostToDevice);cudaMemcpy(device_bias,bias.data(),width*sizeof(__half),cudaMemcpyHostToDevice);
 auto launch=[&](){if(vec)gelu_bias_loop<<<dim3(height,batch),128>>>(src,device_bias,width,height);else gelu_bias_loop_base<<<dim3(height,batch),128>>>(src,device_bias,width,height);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemcpy(src,host.data(),n*sizeof(__half),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),src,n*sizeof(__half),cudaMemcpyDeviceToHost);double maxabs=0,maxrel=0;bool ok=status==cudaSuccess;std::vector<float>flat(n);
  for(size_t i=0;i<n;i++){flat[i]=__half2float(got[i]);double ref=__half2float(expected[i]),e=fabs((double)flat[i]-ref);maxabs=fmax(maxabs,e);maxrel=fmax(maxrel,e/fmax(fabs(ref),1e-3));ok &=std::isfinite(flat[i])&&e<=1e-3;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(flat.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",n,maxabs,maxrel,repeats,filename,n);
 }cudaFree(src);cudaFree(device_bias);puts(allok?"PASS":"FAIL");return allok?0:1;
}
