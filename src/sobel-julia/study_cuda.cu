#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include "../sobel-cuda/sobel.h"
#include "../sobel-cuda/kernels.cu"
#include "../sobel-cuda/reference.cu"
#define CK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(int argc,char**argv){int repeats=argc>1?atoi(argv[1]):100,w=512,h=512,n=w*h;std::vector<uchar4> input(n),expected(n),got(n);uint64_t state=0x1234567;auto bytes=(unsigned char*)input.data();for(int i=0;i<4*n;i++){state=state*6364136223846793005ULL+1442695040888963407ULL;bytes[i]=(state>>32)&255;}
reference(expected.data(),input.data(),w,h,4);uchar4 *din,*dout;CK(cudaMalloc(&din,n*4));CK(cudaMalloc(&dout,n*4));CK(cudaMemcpy(din,input.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemset(dout,0,n*4));sobel_filter<<<dim3(32,32),dim3(16,16)>>>(din,dout,w,h);CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){CK(cudaMemset(dout,0,n*4));CK(cudaDeviceSynchronize());auto t0=std::chrono::steady_clock::now();for(int k=0;k<repeats;k++)sobel_filter<<<dim3(32,32),dim3(16,16)>>>(din,dout,w,h);CK(cudaDeviceSynchronize());double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/repeats;CK(cudaMemcpy(got.data(),dout,n*4,cudaMemcpyDeviceToHost));int ma=0;for(int i=0;i<4*n;i++)ma=std::max(ma,abs(int(((unsigned char*)got.data())[i])-int(((unsigned char*)expected.data())[i])));bool ok=ma==0;all&=ok;char file[100];sprintf(file,"sobel_trial%d.bin",trial);FILE*f=fopen(file,"wb");fwrite(got.data(),1,n*4,f);fclose(f);
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sobel_kernel_only_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,t,ok?"true":"false",n*4,ma,repeats,file,n*4);}
CK(cudaFree(din));CK(cudaFree(dout));puts(all?"PASS":"FAIL");return all?0:1;}
