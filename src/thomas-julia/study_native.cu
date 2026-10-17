#include "../thomas-cuda/cuThomasBatch.cu"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
struct Case{int m,n;std::vector<double> l,d,u,r,x,expected;double *dl,*dd,*du,*dr;};
void cpu(Case&c,std::vector<double>&u,std::vector<double>&r){
 for(int i=0;i<c.n;i++){
  u[i]/=c.d[i];r[i]/=c.d[i];int last=(c.m-1)*c.n+i;
  for(int k=i+c.n;k<last;k+=c.n){double den=c.d[k]-c.l[k]*u[k-c.n];u[k]/=den;r[k]=(r[k]-c.l[k]*r[k-c.n])/den;}
  r[last]=(r[last]-c.l[last]*r[last-c.n])/(c.d[last]-c.l[last]*u[last-c.n]);
  for(int k=last-c.n;k>=i;k-=c.n)r[k]-=u[k]*r[k+c.n];
 }
}
int main(){std::vector<Case>cs;std::vector<double>prefix;const int steps=3;
 for(auto mn:std::vector<std::pair<int,int>>{{2,5},{17,257},{65,33}}){Case c;c.m=mn.first;c.n=mn.second;int N=c.m*c.n;c.l.resize(N);c.d.resize(N);c.u.resize(N);c.r.resize(N);c.x.resize(N);
  for(int j=0;j<c.m;j++)for(int i=0;i<c.n;i++){int k=j*c.n+i;c.l[k]=((i+j)%5-2)/16.;c.u[k]=((2*i+j)%7-3)/16.;c.d[k]=4+((i+3*j)%5)/8.;c.x[k]=((i+2*j)%11-5)/8.;}
  for(int j=0;j<c.m;j++)for(int i=0;i<c.n;i++){int k=j*c.n+i;c.r[k]=c.d[k]*c.x[k];if(j)c.r[k]+=c.l[k]*c.x[k-c.n];if(j<c.m-1)c.r[k]+=c.u[k]*c.x[k+c.n];}
  for(auto v:{c.l,c.d,c.u,c.r})prefix.insert(prefix.end(),v.begin(),v.end());
  auto u=c.u,r=c.r;
  for(int step=0;step<steps;step++){auto ou=u,orr=r;cpu(c,u,r);double res=0;
   for(int j=0;j<c.m;j++)for(int i=0;i<c.n;i++){int k=j*c.n+i;double z=c.d[k]*r[k];if(j)z+=c.l[k]*r[k-c.n];if(j<c.m-1)z+=ou[k]*r[k+c.n];res=std::max(res,std::abs(z-orr[k]));if(step==0 && std::abs(r[k]-c.x[k])>1e-11)return 3;}
   if(res>1e-11)return 4;
  }
  c.expected=u;c.expected.insert(c.expected.end(),r.begin(),r.end());size_t bytes=N*8;
  CK(cudaMalloc(&c.dl,bytes));CK(cudaMalloc(&c.dd,bytes));CK(cudaMalloc(&c.du,bytes));CK(cudaMalloc(&c.dr,bytes));CK(cudaMemcpy(c.dl,c.l.data(),bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(c.dd,c.d.data(),bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(c.du,c.u.data(),bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(c.dr,c.r.data(),bytes,cudaMemcpyHostToDevice));
  cuThomasBatch<<<c.n/128+1,128>>>(c.dl,c.dd,c.du,c.dr,c.m,c.n);CK(cudaGetLastError());cs.push_back(c);
 }CK(cudaDeviceSynchronize());bool all=true;
 for(int t=1;t<=5;t++){
  for(auto&c:cs){CK(cudaMemcpy(c.du,c.u.data(),c.u.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(c.dr,c.r.data(),c.r.size()*8,cudaMemcpyHostToDevice));}CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
  for(auto&c:cs)for(int step=0;step<steps;step++)cuThomasBatch<<<c.n/128+1,128>>>(c.dl,c.dd,c.du,c.dr,c.m,c.n);
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/steps;
  auto out=prefix;bool ok=true;double err=0;size_t count=0;
  for(auto&c:cs){std::vector<double>v(c.u.size()*2);CK(cudaMemcpy(v.data(),c.du,c.u.size()*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(v.data()+c.u.size(),c.dr,c.u.size()*8,cudaMemcpyDeviceToHost));for(size_t k=0;k<v.size();k++){err=std::max(err,std::abs(v[k]-c.expected[k]));ok=ok&&std::isfinite(v[k])&&std::abs(v[k]-c.expected[k])<=1e-10;}count+=v.size();out.insert(out.end(),v.begin(),v.end());}
  all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(t)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 5;fwrite(out.data(),8,out.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,sec,ok?"true":"false",count,err,steps,path.c_str(),out.size());
 }
 for(auto&c:cs){cudaFree(c.dl);cudaFree(c.dd);cudaFree(c.du);cudaFree(c.dr);}puts(all?"PASS":"FAIL");return all?0:1;
}
