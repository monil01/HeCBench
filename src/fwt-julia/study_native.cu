#define main original_hecbench_main
#include "../fwt-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int n=65536,k=128,repeats=1;std::vector<float> host(n),padded(n,0.f),expected(n);for(int i=0;i<n;i++)host[i]=float((i*13+i/31)%31-15)/32.f;for(int j=0;j<k;j++)padded[j]=float((j*7)%17-8)/16.f;dyadicConvolutionCPU(expected.data(),host.data(),padded.data(),16,7);float *data,*kernel;cudaMalloc(&data,n*4);cudaMalloc(&kernel,n*4);cudaMemcpy(data,host.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(kernel,padded.data(),n*4,cudaMemcpyHostToDevice);
 auto launch=[&](){fwtBatchGPU(data,1,16);fwtBatchGPU(kernel,1,16);modulateGPU(data,kernel,n);fwtBatchGPU(data,1,16);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemcpy(data,host.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(kernel,padded.data(),n*4,cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 std::vector<float>got(n);cudaMemcpy(got.data(),data,n*4,cudaMemcpyDeviceToHost);double delta2=0,ref2=0;for(int i=0;i<n;i++){double d=double(got[i])-expected[i];delta2+=d*d;ref2+=double(expected[i])*expected[i];}double l2=sqrt(delta2/ref2);printf("STUDY_NATIVE_RELATIVE_L2 %.12g\n",l2);if(!std::isfinite(l2)||l2>=1e-6)return 6;
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-05));ok&=std::isfinite((double)got[j])&&e<=1e-05;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
