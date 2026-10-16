// Fully initialized deterministic study fixture, not original uninitialized malloc input.
#define main hecbench_original_main
#include "../heat2d-cuda/main.cu"
#undef main
#include <vector>
#include <algorithm>
int main(int argc,char**argv){
 int nx=argc>1?atoi(argv[1]):512,ny=argc>2?atoi(argv[2]):512,repeats=argc>3?atoi(argv[3]):100;
 if(nx<=0||ny<=0||nx%16||ny%16||repeats<=0)return 2;
 int n=nx*ny;std::vector<float>input(n),expected(n),tmp(n),got(n);
 for(int i=0;i<n;i++)input[i]=float((i*13+7)%101)/100.f;
 float sigma=.01f,delta=sigma/(1.f+4.f*sigma),norm=1.f/(1.f+4.f*sigma);expected=input;
 for(int i=0;i<repeats;i++){reference(tmp.data(),expected.data(),delta,norm,nx,ny);expected.swap(tmp);}
 float *a,*b;cudaMalloc(&a,n*sizeof(float));cudaMalloc(&b,n*sizeof(float));
 cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);
 dev_lapl_iter<<<n/256,256>>>(b,a,delta,norm,nx,ny);cudaDeviceSynchronize();
 bool allok=true;
 for(int t=1;t<=5;t++){
  cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<repeats;i++){dev_lapl_iter<<<n/256,256>>>(b,a,delta,norm,nx,ny);std::swap(a,b);}
  auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
  cudaMemcpy(got.data(),a,n*sizeof(float),cudaMemcpyDeviceToHost);
  double maxabs=0,maxrel=0;bool ok=status==cudaSuccess;
  for(int i=0;i<n;i++){double e=fabs(double(got[i])-expected[i]);maxabs=std::max(maxabs,e);maxrel=std::max(maxrel,e/std::max(fabs(double(expected[i])),double(1.1920929e-7f)));ok &= std::isfinite(got[i]) && e<=1.e-5+1.e-5*fabs(double(expected[i]));}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"heat2d_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(got.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",n,maxabs,maxrel,repeats,file,n);
 }
 cudaFree(a);cudaFree(b);puts(allok?"PASS":"FAIL");return allok?0:1;
}
