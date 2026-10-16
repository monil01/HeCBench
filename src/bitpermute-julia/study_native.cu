#define main original_hecbench_main
#include "../bitpermute-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 int lg=argc>1?atoi(argv[1]):15,repeats=argc>2?atoi(argv[2]):11;if(lg<1||lg>20||repeats<1)return 2;size_t n=(size_t)1<<lg;
 std::vector<fr_t>host(n),expected(n),got(n);for(size_t i=0;i<n;i++)host[i]=(fr_t)(i*0x100000001ULL);expected=host;
 if(repeats%2)bit_rev_cpu(expected.data(),host.data(),lg);
 fr_t*d;if(cudaMalloc(&d,n*sizeof(fr_t))!=cudaSuccess)return 3;cudaMemcpy(d,host.data(),n*sizeof(fr_t),cudaMemcpyHostToDevice);bit_rev(d,d,lg);if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemcpy(d,host.data(),n*sizeof(fr_t),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)bit_rev(d,d,lg);auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),d,n*sizeof(fr_t),cudaMemcpyDeviceToHost);long long maxabs=0;bool ok=status==cudaSuccess;for(size_t i=0;i<n;i++){long long e=llabs((long long)got[i]-expected[i]);if(e>maxabs)maxabs=e;ok &=e==0;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(fr_t),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%lld,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",n,maxabs,repeats,filename,n);
 }cudaFree(d);puts(allok?"PASS":"FAIL");return allok?0:1;
}
