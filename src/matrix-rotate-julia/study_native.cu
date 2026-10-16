// Original native kernel unchanged, run under declared deterministic study contract.
#define main original_hecbench_main
#include "../matrix-rotate-cuda/main.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 const int n=argc>1?atoi(argv[1]):1024, repeats=argc>2?atoi(argv[2]):10;
 if(n<2 || (long long)n*n>16777216 || repeats<1)return 2;
 const size_t count=(size_t)n*n;
 std::vector<float> host(count),expected(count),got(count);
 for(size_t i=0;i<count;i++)host[i]=(float)i;expected=host;
 for(int j=0;j<repeats;j++)rotate_matrix_serial(expected.data(),n);
 float*d=nullptr;if(cudaMalloc(&d,count*sizeof(float))!=cudaSuccess)return 3;
 cudaMemcpy(d,host.data(),count*sizeof(float),cudaMemcpyHostToDevice);
 rotate_matrix_parallel<<<(n/2+255)/256,256>>>(d,n);if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;
 for(int trial=1;trial<=5;trial++){
  cudaMemcpy(d,host.data(),count*sizeof(float),cudaMemcpyHostToDevice);cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
  for(int j=0;j<repeats;j++)rotate_matrix_parallel<<<(n/2+255)/256,256>>>(d,n);
  auto status=cudaDeviceSynchronize();
  double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),d,count*sizeof(float),cudaMemcpyDeviceToHost);
  double maxabs=0;bool ok=status==cudaSuccess;
  for(size_t i=0;i<count;i++){double e=fabs((double)got[i]-expected[i]);maxabs=fmax(maxabs,e);ok &= std::isfinite(got[i])&&e==0;}
  allok &=ok;char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);
  FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),count,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",count,maxabs,repeats,filename,count);
 }
 cudaFree(d);puts(allok?"PASS":"FAIL");return allok?0:1;
}
