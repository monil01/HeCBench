// Study uses original tiled CUDA kernel and original independent CPU reference.
#include "../fdtd3d-cuda/FDTD3dGPU.cu"
#include "../fdtd3d-cuda/FDTD3dReference.cu"
#include <vector>
#include <cstdint>
#include <cmath>
int main(int argc,char**argv){
 int nx=argc>1?atoi(argv[1]):96,ny=argc>2?atoi(argv[2]):96,nz=argc>3?atoi(argv[3]):96,steps=argc>4?atoi(argv[4]):5,r=4;
 if(nx<96||ny<96||nz<96||nx%32||ny%8||steps<=0)return 2;
 int n=(nx+8)*(ny+8)*(nz+8),padding=28;std::vector<float> input(n),expected(n),got(n),coeff(5,.1f);uint32_t s=1;
 for(int i=0;i<n;i++){s=s*1103515245u+12345u;input[i]=float((s>>8)&0xffffffu)/16777216.f;}
 fdtdReference(expected.data(),input.data(),coeff.data(),nx,ny,nz,r,steps);
 float *a,*b,*c;cudaMalloc(&a,(n+padding)*sizeof(float));cudaMalloc(&b,(n+padding)*sizeof(float));cudaMalloc(&c,coeff.size()*sizeof(float));cudaMemcpy(c,coeff.data(),coeff.size()*sizeof(float),cudaMemcpyHostToDevice);
 dim3 block(32,8),grid(nx/32,ny/8);
 cudaMemset(a,0,(n+padding)*sizeof(float));cudaMemset(b,0,(n+padding)*sizeof(float));cudaMemcpy(a+padding,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b+padding,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);
 finite_difference<<<grid,block>>>(b,a,c,nx,ny,nz,padding);cudaDeviceSynchronize();bool allok=true;
 for(int t=1;t<=5;t++){
  cudaMemcpy(a+padding,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b+padding,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
  for(int j=0;j<steps;j++){finite_difference<<<grid,block>>>(b,a,c,nx,ny,nz,padding);std::swap(a,b);}
  auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/steps;
  cudaMemcpy(got.data(),a+padding,n*sizeof(float),cudaMemcpyDeviceToHost);double ma=0,mr=0;bool ok=status==cudaSuccess;
  for(int i=0;i<n;i++){double e=fabs(double(got[i])-expected[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(double(expected[i])),double(1.1920929e-7f)));ok &= std::isfinite(got[i])&&e<=1.e-5+1.e-4*fabs(double(expected[i]));}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"fdtd3d_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(got.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",n,ma,mr,steps,file,n);
 }
 cudaFree(a);cudaFree(b);cudaFree(c);puts(allok?"PASS":"FAIL");return allok?0:1;
}
