#include "../triad-cuda/triad.cu"
#include <vector>
#include <fstream>
#include <chrono>
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4194304,repeats=argc>2?atoi(argv[2]):100;if(n<=0||n%128||repeats<=0)return 2;
 std::vector<float>a(n),b(n),v(n);for(int i=0;i<n;i++){a[i]=float(i%1009)/100.f;b[i]=float(i%1013)/100.f;}
 float *da,*db,*dc;cudaMalloc(&da,n*4);cudaMalloc(&db,n*4);cudaMalloc(&dc,n*4);cudaMemcpy(da,a.data(),n*4,cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),n*4,cudaMemcpyHostToDevice);
 triad<<<n/128,128>>>(da,db,dc,1.75f);cudaDeviceSynchronize();bool allok=true;
 for(int t=1;t<=5;t++){
  cudaMemset(dc,0,n*4);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int r=0;r<repeats;r++)triad<<<n/128,128>>>(da,db,dc,1.75f);cudaDeviceSynchronize();double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(v.data(),dc,n*4,cudaMemcpyDeviceToHost);bool ok=true;double emax=0;for(int i=0;i<n;i++){double e=fabs(v[i]-(a[i]+1.75f*b[i]));emax=fmax(emax,e);ok&=std::isfinite(v[i])&&e<=1e-3;}ok&=cudaGetLastError()==cudaSuccess;allok&=ok;
  std::string file=std::string(getenv("STUDY_OUTPUT_DIR"))+"/triad_"+std::to_string(t)+".bin";std::ofstream out(file,std::ios::binary);out.write((char*)v.data(),n*4);out.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.9g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,sec,ok?"true":"false",n,emax,repeats,file.c_str(),n);
 }
 cudaFree(da);cudaFree(db);cudaFree(dc);return allok?0:1;
}
