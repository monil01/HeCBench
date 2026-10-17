#define main original_iso2dfd_main
#include "../iso2dfd-cuda/iso2dfd.cu"
#undef main
#include <vector>
#include <algorithm>
#include <cstdlib>
#define CHECK(call) do { cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;} }while(0)
int main(){
 const size_t nr=128,nc=160,n=nr*nc;const int steps=24;const float coef=(DT*DT)/(DXY*DXY);
 std::vector<float> init(n),zero(n),vel(n);initialize(init.data(),zero.data(),vel.data(),nr,nc);
 auto refcur=init,refold=zero;iso_2dfd_iteration_cpu(refold.data(),refcur.data(),vel.data(),coef,nr,nc,steps);
 float *a,*b,*v;CHECK(cudaMalloc(&a,n*4));CHECK(cudaMalloc(&b,n*4));CHECK(cudaMalloc(&v,n*4));CHECK(cudaMemcpy(v,vel.data(),n*4,cudaMemcpyHostToDevice));
 dim3 threads(16,16),blocks((nc+15)/16,(nr+15)/16);
 CHECK(cudaMemcpy(a,init.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,zero.data(),n*4,cudaMemcpyHostToDevice));
 iso_2dfd_kernel<<<blocks,threads>>>(b,a,v,coef,nr,nc);CHECK(cudaDeviceSynchronize());
 bool all=true;
 for(int trial=1;trial<=5;trial++){
  CHECK(cudaMemcpy(a,init.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,zero.data(),n*4,cudaMemcpyHostToDevice));CHECK(cudaDeviceSynchronize());
  float* cur=a;float* old=b;auto start=std::chrono::steady_clock::now();
  for(int step=0;step<steps;step++){iso_2dfd_kernel<<<blocks,threads>>>(old,cur,v,coef,nr,nc);std::swap(cur,old);}
  CHECK(cudaDeviceSynchronize());auto end=std::chrono::steady_clock::now();double sec=std::chrono::duration<double>(end-start).count()/steps;
  std::vector<float> out(2*n);CHECK(cudaMemcpy(out.data(),cur,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(out.data()+n,old,n*4,cudaMemcpyDeviceToHost));
  bool ok=true;double ma=0,mr=0;
  for(size_t i=0;i<2*n;i++){double expected=(i<n?refcur[i]:refold[i-n]);double err=fabs(double(out[i])-expected);ok=ok&&std::isfinite(out[i])&&std::isfinite(expected)&&err<=1e-4+1e-5*fabs(expected);ma=std::max(ma,err);mr=std::max(mr,err/std::max(1e-4,fabs(expected)));}
  all=all&&ok;std::string name="output_trial"+std::to_string(trial)+".bin";const char* dir=std::getenv("STUDY_OUTPUT_DIR");if(dir)name=std::string(dir)+"/"+name;
  FILE* f=fopen(name.c_str(),"wb");if(!f)return 3;fwrite(out.data(),4,out.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",out.size(),ma,mr,steps,name.c_str(),out.size());
 }
 cudaFree(a);cudaFree(b);cudaFree(v);puts(all?"PASS":"FAIL");return all?0:1;
}
