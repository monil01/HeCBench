// Only int4/aligned-length configuration; original tail/default issues retained.
#define main original_hecbench_main
#include "../threadcpy-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
int main(int argc,char**argv){
 size_t n=argc>1?strtoull(argv[1],nullptr,10):1048576;int repeats=argc>2?atoi(argv[2]):10;if(n<1||n%1024||repeats<1)return 2;
 std::vector<int>host(n),got(n);for(size_t i=0;i<n;i++)host[i]=(int)(i%101)-50;
 int*input,*out;GPU_CHECK(cudaMalloc(&input,n*sizeof(int)));GPU_CHECK(cudaMalloc(&out,n*sizeof(int)));GPU_CHECK(cudaMemcpy(input,host.data(),n*sizeof(int),cudaMemcpyHostToDevice));
 auto launch=[&](){threads_copy_kernel<int,4><<<(n+1023)/1024,256>>>(input,out,n);};launch();GPU_CHECK(cudaDeviceSynchronize());
 bool allok=true;for(int trial=1;trial<=5;trial++){
  GPU_CHECK(cudaMemset(out,0,n*sizeof(int)));GPU_CHECK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  GPU_CHECK(cudaMemcpy(got.data(),out,n*sizeof(int),cudaMemcpyDeviceToHost));long long maxabs=0;bool ok=status==cudaSuccess;for(size_t i=0;i<n;i++){long long e=llabs((long long)got[i]-host[i]);if(e>maxabs)maxabs=e;ok &=e==0;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(int),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%lld,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",n,maxabs,repeats,filename,n);
 }GPU_CHECK(cudaFree(input));GPU_CHECK(cudaFree(out));puts(allok?"PASS":"FAIL");return allok?0:1;
}
