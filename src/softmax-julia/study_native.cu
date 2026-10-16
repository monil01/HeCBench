// Mode0/naive only; optimized cooperative_groups softMax2 not measured.
#define main original_hecbench_main
#include "../softmax-cuda/main.cu"
#undef main
#include <vector>
#include <cmath>
int main(int argc,char**argv){
 int rows=argc>1?atoi(argv[1]):8192,cols=argc>2?atoi(argv[2]):128,repeats=argc>3?atoi(argv[3]):10;if(rows<1||cols<1||repeats<1)return 2;size_t n=(size_t)rows*cols;
 std::vector<float>host(n),expected(n),got(n);for(size_t i=0;i<n;i++)host[i]=(float)((17*i)%13);softMax_cpu(rows,cols,host.data(),expected.data());
 float*a,*out;if(cudaMalloc(&a,n*sizeof(float))!=cudaSuccess||cudaMalloc(&out,n*sizeof(float))!=cudaSuccess)return 3;cudaMemcpy(a,host.data(),n*sizeof(float),cudaMemcpyHostToDevice);
 auto launch=[&](){softMax<<<(rows+255)/256,256>>>(rows,cols,a,out);};launch();if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
 bool allok=true;for(int trial=1;trial<=5;trial++){
  cudaMemset(out,0,n*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int j=0;j<repeats;j++)launch();auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),out,n*sizeof(float),cudaMemcpyDeviceToHost);double maxabs=0,maxrel=0;bool ok=status==cudaSuccess;for(size_t i=0;i<n;i++){double e=fabs((double)got[i]-expected[i]);maxabs=fmax(maxabs,e);maxrel=fmax(maxrel,e/fmax(fabs((double)expected[i]),1e-3));ok &=std::isfinite(got[i])&&e<=1e-3;}allok &=ok;
  char filename[80];snprintf(filename,sizeof(filename),"output_trial%d.bin",trial);FILE*f=fopen(filename,"wb");if(!f)return 5;fwrite(got.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",n,maxabs,maxrel,repeats,filename,n);
 }cudaFree(a);cudaFree(out);puts(allok?"PASS":"FAIL");return allok?0:1;
}
