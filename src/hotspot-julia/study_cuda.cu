// Shared deterministic synthetic fixture; original calc_temp device kernel retained.
#define main hecbench_original_main
#include "../hotspot-cuda/hotspot.cu"
#undef main
#include <vector>
int main(int argc,char**argv){
 int steps=argc>1?atoi(argv[1]):200,L=512,n=L*L;if(steps<=0)return 2;
 std::vector<float>input(n),power(n),expected(n),got(n);for(int i=0;i<n;i++){input[i]=300.f+float(i%101)*.5f;power[i]=float(i%17)*.03125f;}
 float grid=.016f/float(L),cap=(((.5f*1.75e6f)*.0005f)*grid)*grid,rx=grid/(((2.f*100.f)*.0005f)*grid),ry=rx,rz=.0005f/((100.f*grid)*grid),step=.001f/(3.e6f/((.5f*.0005f)*1.75e6f));
 stencil(steps,power.data(),input.data(),expected.data(),L,L,cap,rx,ry,rz,step);
 float*a,*b,*p;cudaMalloc(&a,n*sizeof(float));cudaMalloc(&b,n*sizeof(float));cudaMalloc(&p,n*sizeof(float));cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(p,power.data(),n*sizeof(float),cudaMemcpyHostToDevice);
 dim3 threads(16,16),blocks((L+13)/14,(L+13)/14);calc_temp<<<blocks,threads>>>(1,p,a,b,L,L,1,1,cap,rx,ry,rz,step);cudaDeviceSynchronize();bool allok=true;
 for(int t=1;t<=5;t++){
  cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b,input.data(),n*sizeof(float),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
  for(int j=0;j<steps;j++){calc_temp<<<blocks,threads>>>(1,p,a,b,L,L,1,1,cap,rx,ry,rz,step);std::swap(a,b);}
  auto status=cudaDeviceSynchronize();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/steps;cudaMemcpy(got.data(),a,n*sizeof(float),cudaMemcpyDeviceToHost);
  double ma=0,mr=0;bool ok=status==cudaSuccess;for(int i=0;i<n;i++){double e=fabs(double(got[i])-expected[i]);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(double(expected[i])),double(1.1920929e-7f)));ok &=std::isfinite(got[i])&&e<=1.e-3;}
  allok &=ok;char file[80];snprintf(file,sizeof(file),"hotspot_trial%d.bin",t);FILE*f=fopen(file,"wb");fwrite(got.data(),sizeof(float),n,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok?"true":"false",n,ma,mr,steps,file,n);
 }
 cudaFree(a);cudaFree(b);cudaFree(p);puts(allok?"PASS":"FAIL");return allok?0:1;
}
