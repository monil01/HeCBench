#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "../expdist-cuda/kernel.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(3);}}while(0)
std::vector<double> load(const char*n){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f)exit(4);std::vector<double>v(f.tellg()/8);f.seekg(0);f.read((char*)v.data(),v.size()*8);return v;}
template<class T>struct Buf{std::vector<T>h;T*d;Buf(const std::vector<double>&v,int offset,int n):h(n+32,T(-777.25)){for(int i=0;i<n;i++)h[16+i]=T(v[offset+i]);CK(cudaMalloc(&d,h.size()*sizeof(T)));CK(cudaMemcpy(d,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}T*ptr(){return d+16;}std::vector<T>get(){std::vector<T>v(h.size());CK(cudaMemcpy(v.data(),d,v.size()*sizeof(T),cudaMemcpyDeviceToHost));return v;}~Buf(){cudaFree(d);}};
template<class T>void append(std::vector<double>&dst,const std::vector<T>&v){for(auto x:v)dst.push_back(double(x));}
template<class T>void cases(int trial,const std::vector<double>&input,const std::vector<double>&gold,int&io,int&go,std::vector<double>&prefix,std::vector<double>&outputs,std::vector<double>&times,bool&ok,double&maxerr,double&maxfrac){
 for(auto shape:std::vector<std::pair<int,int>>{{128,32},{256,64},{256,256}})for(int fixture=0;fixture<3;fixture++){
  int m=shape.first,n=shape.second,nb=(m/128)*(n/32);Buf<T>A(input,io,2*m);io+=2*m;Buf<T>B(input,io,2*n);io+=2*n;Buf<T>sa(input,io,m);io+=m;Buf<T>sb(input,io,n);io+=n;std::vector<double>si(nb,-999.5),so(1,-999.5);Buf<T>p(si,0,nb),o(so,0,1);dim3 grid(m/128,n/32),threads(32,8);
  auto launch=[&](){distance<T><<<grid,threads>>>(A.ptr(),B.ptr(),m,n,sa.ptr(),sb.ptr(),p.ptr());reduce_cross_term<T><<<1,256>>>(o.ptr(),p.ptr(),m,n,nb);};
  if(trial==0){launch();CK(cudaDeviceSynchronize());go+=nb+1;continue;}
  auto bp=p.get(),bo=o.get();ok=ok&&bp==p.h&&bo==o.h;CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<10;rep++)launch();CK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/10);
  for(auto x:{&A,&B,&sa,&sb}){auto a=x->get();ok=ok&&a==x->h;append(prefix,a);}append(prefix,bp);append(prefix,bo);auto ap=p.get(),ao=o.get();for(int j=0;j<16;j++)ok=ok&&ap[j]==T(-777.25)&&ap[nb+16+j]==T(-777.25)&&ao[j]==T(-777.25)&&ao[17+j]==T(-777.25);
  for(int k=0;k<=nb;k++){double v=k<nb?ap[k+16]:ao[16],ref=gold[go+k],err=fabs(v-ref),budget=(sizeof(T)==4?.001:1e-9)+(sizeof(T)==4?1e-5:1e-12)*fabs(ref);ok=ok&&std::isfinite(v)&&err<=budget;maxerr=std::max(maxerr,err);maxfrac=std::max(maxfrac,err/budget);if(fixture==2)ok=ok&&v==(k<nb?4096.:double(m)*n);}go+=nb+1;append(outputs,ap);append(outputs,ao);
 }
}
int main(){auto input=load("inputs_f64.bin"),gold=load("oracle_f64.bin");bool all=true;for(int trial=0;trial<=5;trial++){int io=0,go=0;bool ok=true;double maxerr=0,maxfrac=0;std::vector<double>prefix,outputs,times;cases<float>(trial,input,gold,io,go,prefix,outputs,times,ok,maxerr,maxfrac);cases<double>(trial,input,gold,io,go,prefix,outputs,times,ok,maxerr,maxfrac);if(trial==0)continue;all=all&&ok;prefix.insert(prefix.end(),outputs.begin(),outputs.end());const char*dir=getenv("STUDY_OUTPUT_DIR");std::string path=std::string(dir?dir:".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);double total=0;for(auto t:times)total+=t;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"max_oracle_budget_fraction\":%.12g,\"timing_region\":\"sum18_pairwise_distance_reduction_hostwindows_R10\",\"jit_excluded\":true,\"path_times_seconds\":[",trial,total,ok?"true":"false",prefix.size(),maxerr,maxfrac);for(size_t i=0;i<times.size();i++)printf("%s%.12g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),prefix.size());}puts(all?"PASS":"FAIL");return all?0:2;}
