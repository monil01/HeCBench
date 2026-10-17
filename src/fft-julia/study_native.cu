#define main original_hecbench_main
#include "../fft-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
#include <chrono>
int main(int argc,char**argv){
const int nfft=128,n=nfft*512,repeats=10;std::vector<T2>input(n),roundtrip(n),freq(n),spectrum(n),source_cpu(n);std::vector<double>got(4*n),expected(4*n);T2 twiddle[512];for(int j=0;j<512;j++)twiddle[j]=exp_i(-2*M_PI*j/512.);for(int f=0;f<nfft;f++)for(int j=0;j<512;j++)input[f*512+j]={(double)((17*j+13*f)%257-128)/128.,(double)((29*j+7*f)%251-125)/128.};
for(int f=0;f<nfft;f++)for(int k=0;k<512;k++){T2 sum={0,0};for(int j=0;j<512;j++)sum=cmplx_add(sum,cmplx_mul(input[f*512+j],twiddle[k*j%512]));spectrum[f*512+k]=sum;}
source_cpu=input;fft1D_512_reference<64>(source_cpu.data(),nfft);for(int i=0;i<n;i++)if(fabs(source_cpu[i].x-spectrum[i].x)>1e-6||fabs(source_cpu[i].y-spectrum[i].y)>1e-6)return 6;
for(int i=0;i<n;i++){expected[2*i]=spectrum[i].x;expected[2*i+1]=spectrum[i].y;expected[2*n+2*i]=input[i].x;expected[2*n+2*i+1]=input[i].y;}
T2*data;cudaMalloc(&data,n*sizeof(T2));cudaMemcpy(data,input.data(),n*sizeof(T2),cudaMemcpyHostToDevice);

 auto launch=[&](){fft1D_512<<<nfft,64>>>(data);ifft1D_512<<<nfft,64>>>(data);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
 cudaMemcpy(data,input.data(),n*sizeof(T2),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
 cudaMemcpy(roundtrip.data(),data,n*sizeof(T2),cudaMemcpyDeviceToHost);fft1D_512<<<nfft,64>>>(data);auto validation_status=cudaDeviceSynchronize();if(validation_status!=cudaSuccess)return 7;cudaMemcpy(freq.data(),data,n*sizeof(T2),cudaMemcpyDeviceToHost);double original_energy=0,fft_energy=0;for(int i=0;i<n;i++){got[2*i]=freq[i].x;got[2*i+1]=freq[i].y;got[2*n+2*i]=roundtrip[i].x;got[2*n+2*i+1]=roundtrip[i].y;original_energy+=input[i].x*input[i].x+input[i].y*input[i].y;fft_energy+=freq[i].x*freq[i].x+freq[i].y*freq[i].y;}double ee=fabs(fft_energy-512*original_energy)/(512*original_energy);if(!(ee<1e-10))return 8;printf("STUDY_FFT_ACCURACY forward_and_roundtrip_all_components=true energy_relative_error=%.12g storage=ComplexF64 native_tolerance=1e-6\n",ee);
 bool ok=status==cudaSuccess;double error=0,relative=0;for(size_t j=0;j<got.size();j++){double e=fabs((double)got[j]-expected[j]);error=fmax(error,e);relative=fmax(relative,e/fmax(fabs((double)expected[j]),1e-06));ok&=std::isfinite((double)got[j])&&e<=1e-06;}allok&=ok;
 char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(double),got.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",got.size(),error,relative,repeats,filename,got.size());
 }return allok?0:1;
}
