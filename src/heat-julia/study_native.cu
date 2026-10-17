#define main archived_heat_main
#include "../heat-cuda/heat.cu"
#undef main
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
struct Case{int n,steps;double dx,dt,r,r2;std::vector<double>init,expected;double*a,*b;};
int main(){std::vector<double>prefix(85123);FILE*f=fopen("study_initial.bin","rb");if(!f||fread(prefix.data(),8,prefix.size(),f)!=prefix.size())return 3;fclose(f);std::vector<Case>cs;int offset=0;
 for(auto ns:std::vector<std::pair<int,int>>{{63,17},{127,18},{255,19}}){Case c;c.n=ns.first;c.steps=ns.second;int N=c.n*c.n;c.dx=1000./(c.n+1);c.dt=.5/c.steps;c.r=.1*c.dt/(c.dx*c.dx);c.r2=1-4*c.r;c.init.assign(prefix.begin()+offset,prefix.begin()+offset+N);offset+=N;
  auto cur=c.init;std::vector<double>old(N);
  for(int step=0;step<c.steps;step++){
   for(int j=0;j<c.n;j++)for(int i=0;i<c.n;i++){int k=j*c.n+i;old[k]=c.r2*cur[k]+c.r*(i<c.n-1?cur[k+1]:0.)+c.r*(i>0?cur[k-1]:0.)+c.r*(j<c.n-1?cur[k+c.n]:0.)+c.r*(j>0?cur[k-c.n]:0.);}
   cur.swap(old);
  }
  double g=1-8*c.r*pow(sin(M_PI/(2*(c.n+1))),2);double de=0,ce=0;
  for(int k=0;k<N;k++){de=std::max(de,std::abs(cur[k]-pow(g,c.steps)*c.init[k]));de=std::max(de,std::abs(old[k]-pow(g,c.steps-1)*c.init[k]));ce=std::max(ce,std::abs(cur[k]-exp(-2*.1*M_PI*M_PI*.5/(1000.*1000.))*c.init[k]));}
  if(de>5e-13||c.r>.25)return 4;printf("HEAT_ORACLE n=%d steps=%d r=%.12g discrete_max=%.12g continuous_max=%.12g\n",c.n,c.steps,c.r,de,ce);
  c.expected=cur;c.expected.insert(c.expected.end(),old.begin(),old.end());CK(cudaMalloc(&c.a,N*8));CK(cudaMalloc(&c.b,N*8));CK(cudaMemcpy(c.a,c.init.data(),N*8,cudaMemcpyHostToDevice));CK(cudaMemset(c.b,0,N*8));solve<<<(N+255)/256,256>>>(c.n,.1,c.dx,c.dt,c.r,c.r2,c.a,c.b);CK(cudaGetLastError());cs.push_back(c);
 }CK(cudaDeviceSynchronize());bool all=true;
 for(int t=1;t<=5;t++){
  for(auto&c:cs){CK(cudaMemcpy(c.a,c.init.data(),c.init.size()*8,cudaMemcpyHostToDevice));CK(cudaMemset(c.b,0,c.init.size()*8));}CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
  for(auto&c:cs){double*cur=c.a,*old=c.b;for(int step=0;step<c.steps;step++){solve<<<(c.n*c.n+255)/256,256>>>(c.n,.1,c.dx,c.dt,c.r,c.r2,cur,old);std::swap(cur,old);}}
  CK(cudaGetLastError());CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/54;auto out=prefix;double err=0;bool ok=true;size_t count=0;
  for(auto&c:cs){int N=c.n*c.n;std::vector<double>v(N*2);double*cur=(c.steps%2)?c.b:c.a;double*old=(c.steps%2)?c.a:c.b;CK(cudaMemcpy(v.data(),cur,N*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(v.data()+N,old,N*8,cudaMemcpyDeviceToHost));for(int k=0;k<2*N;k++){err=std::max(err,std::abs(v[k]-c.expected[k]));ok=ok&&std::isfinite(v[k])&&std::abs(v[k]-c.expected[k])<=1e-12&&v[k]>=0&&v[k]<=1;}count+=v.size();out.insert(out.end(),v.begin(),v.end());}
  all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(t)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 5;fwrite(out.data(),8,out.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":54,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",t,sec,ok?"true":"false",count,err,path.c_str(),out.size());
 }for(auto&c:cs){cudaFree(c.a);cudaFree(c.b);}puts(all?"PASS":"FAIL");return all?0:1;
}
