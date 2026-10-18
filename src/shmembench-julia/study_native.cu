#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../shmembench-cuda/shmem_kernels.cu"
#define CK(x)do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
int main(int argc,char**argv){const int N=1048576;int repeat=argc>1?atoi(argv[1]):1000;std::ifstream f("expected_f32_guards.bin",std::ios::binary);std::vector<float>expected(2*N);f.read((char*)expected.data(),2*N*4);std::vector<float>out(2*N,16777215.f);float*device;CK(cudaMalloc(&device,N*8));CK(cudaMemcpy(device,out.data(),N*8,cudaMemcpyHostToDevice));auto run=[&](){benchmark_shmem<<<1024,256>>>((float4*)device);CK(cudaGetLastError());};run();CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int r=0;r<repeat;r++)run();CK(cudaDeviceSynchronize());double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;CK(cudaMemcpy(out.data(),device,N*8,cudaMemcpyDeviceToHost));bool ok=out==expected;all &=ok;double err=0;for(int i=0;i<2*N;i++)err=fmax(err,fabs(out[i]-expected[i]));const char*dir=getenv("STUDY_OUTPUT_DIR");char file[4096];snprintf(file,sizeof(file),"%s/shmembench_trial%d.bin",dir?dir:".",trial);std::ofstream of(file,std::ios::binary);of.write((char*)out.data(),N*8);of.close();
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"defined_payload_values\":%d,\"unwritten_guard_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"internal_repeats\":%d,\"jit_excluded\":true,\"timing_region\":\"source1000_kernel_loopmean_1024_swapiterations_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,time,ok?"true":"false",2*N,N,N,err,repeat,file,2*N);fflush(stdout);}
puts(all?"PASS":"FAIL");return all?0:1;}
