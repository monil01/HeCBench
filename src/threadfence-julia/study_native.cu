#include <vector>
#define main hecbench_original_main
#include "threadfence-cuda/main.cu"
#undef main
int main(int argc,char**argv){int repeat=argc>1?atoi(argv[1]):10,n=argc>2?atoi(argv[2]):4096,grids=(n+255)/256;
 std::vector<float>input(n,-1),expected(grids,-256),values(grids+1);expected.back()=-float(n-(grids-1)*256);expected[0]=-float(n);expected.push_back(0);
 float *in,*out;unsigned int*counter;cudaMalloc(&in,n*4);cudaMalloc(&out,grids*4);cudaMalloc(&counter,4);cudaMemcpy(in,input.data(),n*4,cudaMemcpyHostToDevice);cudaMemset(counter,0,4);sum<<<grids,256>>>(in,n,counter,out);cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){cudaMemset(out,0,grids*4);cudaMemset(counter,0,4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeat;j++)sum<<<grids,256>>>(in,n,counter,out);cudaDeviceSynchronize();double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeat;cudaMemcpy(values.data(),out,grids*4,cudaMemcpyDeviceToHost);unsigned int count;cudaMemcpy(&count,counter,4,cudaMemcpyDeviceToHost);values[grids]=float(count);bool correct=values==expected;
 char path[64];snprintf(path,sizeof(path),"trial_%d.bin",trial);FILE*f=fopen(path,"wb");fwrite(values.data(),4,values.size(),f);fclose(f);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"timing_region\":\"synchronized_complete_reduction_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,time,correct?"true":"false",grids+1,repeat,path,grids+1);if(!correct)return 1;
 }cudaFree(in);cudaFree(out);cudaFree(counter);return 0;}
