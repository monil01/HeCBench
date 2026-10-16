#include <vector>
#define main hecbench_original_main
#include "lfib4-cuda/main.cu"
#undef main
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):16384,r=16,s=n/r;std::vector<uint32_t>seed(256),expected(n),values(n);for(int i=0;i<256;i++)expected[i]=seed[i]=uint32_t(i*1664525u+1013904223u);LFIB4(n,expected.data());uint32_t*x,*y;cudaMalloc(&x,n*4);cudaMalloc(&y,768*4);cudaStream_t a,b;cudaStreamCreate(&a);cudaStreamCreate(&b);
 auto reset=[&](){cudaMemset(x,0,n*4);cudaMemcpy(x,seed.data(),256*4,cudaMemcpyHostToDevice);cudaMemset(y,0,768*4);uint32_t one=1;cudaMemcpy(y+512,&one,4,cudaMemcpyHostToDevice);cudaDeviceSynchronize();};
 auto pipeline=[&](){firstColGPU<<<1,256,0,a>>>(x,s);colYGPU<<<1,256,0,b>>>(y,s);cudaStreamSynchronize(a);cudaStreamSynchronize(b);lastEntGPU<<<1,512>>>(x,y,s,r);colsGPU<<<(r+7)/8,256>>>(x,s,r);cudaDeviceSynchronize();};reset();pipeline();
 for(int trial=1;trial<=5;trial++){reset();auto start=std::chrono::steady_clock::now();pipeline();double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();cudaMemcpy(values.data(),x,n*4,cudaMemcpyDeviceToHost);bool correct=values==expected;char path[64];snprintf(path,sizeof(path),"trial_%d.bin",trial);FILE*f=fopen(path,"wb");fwrite(values.data(),4,n,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"timing_region\":\"four_stage_parallel_pipeline_with_stream_joins\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,time,correct?"true":"false",n,correct?0:1,path,n);if(!correct)return 1;}
 cudaFree(x);cudaFree(y);cudaStreamDestroy(a);cudaStreamDestroy(b);return 0;}
