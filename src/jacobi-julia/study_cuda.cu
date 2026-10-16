// Fixed-step study retains original kernel's block reduction plus error reset/copy each step.
#define main hecbench_original_main
#include "../jacobi-cuda/main.cu"
#undef main
#include <vector>
#include <algorithm>
int main(int argc,char**argv){
 int steps=argc>1?atoi(argv[1]):20,n=N*N;if(steps<=0)return 2;
 std::vector<float> input(n,0),expected(n),tmp(n),expectednorms(steps),norms(steps),got(n+steps);
 for(int j=0;j<N;j++)for(int i=0;i<N;i++)if(i==0||i==N-1||j==0||j==N-1)input[i+j*N]=float((i+3*j)%17)/16.f;
 expected=input;tmp=input;
 for(int k=0;k<steps;k++){double total=0;for(int j=1;j<N-1;j++)for(int i=1;i<N-1;i++){int idx=i+j*N;float val=.25f*(expected[idx+1]+expected[idx-1]+expected[idx+N]+expected[idx-N]);tmp[idx]=val;float df=val-expected[idx];total+=double(df*df);}expectednorms[k]=float(sqrt(total/n));expected.swap(tmp);}
 float*a,*b,*err;cudaMalloc(&a,n*sizeof(float));cudaMalloc(&b,n*sizeof(float));cudaMalloc(&err,sizeof(float));cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemset(err,0,sizeof(float));
 dim3 blocks(N/16,N/16),threads(16,16);jacobi_step<<<blocks,threads>>>(b,a,err);cudaDeviceSynchronize();bool allok=true;
 for(int t=1;t<=5;t++){
  cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int k=0;k<steps;k++){cudaMemset(err,0,sizeof(float));jacobi_step<<<blocks,threads>>>(b,a,err);std::swap(a,b);float e;cudaMemcpy(&e,err,sizeof(float),cudaMemcpyDeviceToHost);norms[k]=sqrtf(e/float(n));}
  auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/steps;cudaMemcpy(got.data(),a,n*sizeof(float),cudaMemcpyDeviceToHost);std::copy(norms.begin(),norms.end(),got.begin()+n);
  double ma=0,mr=0;bool ok=status==cudaSuccess;for(int i=0;i<n+steps;i++){float ref=i<n?expected[i]:expectednorms[i-n];double e=fabs(double(got[i])-ref);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(double(ref)),double(1.1920929e-7f)));ok &=std::isfinite(got[i])&&e<=1.e-6+1.e-4*fabs(double(ref));}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"jacobi_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(got.data(),sizeof(float),n+steps,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"fixed_jacobi_step_loop_with_error_reset_reduction_copy_per_step\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",n+steps,ma,mr,steps,file,n+steps);
 }
 cudaFree(a);cudaFree(b);cudaFree(err);puts(allok?"PASS":"FAIL");return allok?0:1;
}
