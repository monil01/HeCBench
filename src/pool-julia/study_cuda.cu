// Average-pooling gradient, NCHW, exclusive=true; immutable original native specialization.
#define main hecbench_original_main
#include "../pool-cuda/main.cu"
#undef main
#include <vector>
#include <algorithm>
int main(int argc,char**argv){
 int repeats=argc>1?atoi(argv[1]):50,B=2,C=8,H=128,W=128,OH=30,OW=30,n=B*C*H*W,m=B*C*OH*OW;if(repeats<=0)return 2;
 std::vector<float>input(n,0),output(m,0),dy(m,float(H*W)),expected(n),got(n);AvgPoolGrad<float> proc;
 reference<AvgPoolGrad<float>,float>(n,input.data(),output.data(),dy.data(),C,H,W,OH,OW,11,11,4,4,1,1,proc,true,expected.data(),false);
 float*a,*b,*g,*dx;cudaMalloc(&a,n*sizeof(float));cudaMalloc(&b,m*sizeof(float));cudaMalloc(&g,m*sizeof(float));cudaMalloc(&dx,n*sizeof(float));cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,output.data(),m*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(g,dy.data(),m*sizeof(float),cudaMemcpyHostToDevice);
 int blocks=(n+255)/256;
 KernelPool2DGrad<AvgPoolGrad<float>,float,11,11,4,4,1,1,true><<<blocks,256>>>(n,a,b,g,C,H,W,OH,OW,proc,dx,false);cudaDeviceSynchronize();bool allok=true;
 for(int t=1;t<=5;t++){
  cudaMemset(dx,0,n*sizeof(float));cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int k=0;k<repeats;k++)KernelPool2DGrad<AvgPoolGrad<float>,float,11,11,4,4,1,1,true><<<blocks,256>>>(n,a,b,g,C,H,W,OH,OW,proc,dx,false);
  auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;cudaMemcpy(got.data(),dx,n*sizeof(float),cudaMemcpyDeviceToHost);
  double ma=0,mr=0;bool ok=status==cudaSuccess;for(int i=0;i<n;i++){double e=fabs(double(got[i])-expected[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(double(expected[i])),double(1.1920929e-7f)));ok &=std::isfinite(got[i])&&e<=1.e-3;}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"pool_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(got.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",n,ma,mr,repeats,file,n);
 }
 cudaFree(a);cudaFree(b);cudaFree(g);cudaFree(dx);puts(allok?"PASS":"FAIL");return allok?0:1;
}
