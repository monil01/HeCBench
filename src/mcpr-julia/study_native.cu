#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../mcpr-cuda/kernels.h"
#include "study_fixture.h"
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Case{int n,M;std::vector<double>A,R,AS,RS,want;std::vector<int64_t>counts;double*da;double*dr;double*ds;double*dt;double*dp[3];};
int main(){std::vector<Case>cases;std::vector<double>prefix;
 for(int id=1;id<=2;++id){Case c;c.n=id==1?193:5;c.M=id==1?97:3;const double*a=id==1?ALPHAS1:ALPHAS2;const double*r=id==1?DRAWS1:DRAWS2;c.A.assign(a,a+c.n*21);c.R.assign(r,r+c.M*21);c.AS.resize(c.A.size());c.RS.resize(c.R.size());for(int i=0;i<c.n;++i)for(int k=0;k<21;++k)c.AS[k*c.n+i]=c.A[i*21+k];for(int i=0;i<c.M;++i)for(int k=0;k<21;++k)c.RS[k*c.M+i]=c.R[i*21+k];c.counts.assign(c.n*21,0);
  for(int i=0;i<c.n;++i)for(int m=0;m<c.M;++m){double values[21];for(int k=0;k<21;++k)values[k]=c.A[i*21+k]+c.R[m*21+k];double maximum=*std::max_element(values,values+21);int best=values[20]==maximum?20:int(std::find(values,values+21,maximum)-values);++c.counts[i*21+best];}for(int64_t v:c.counts)c.want.push_back(double(v)/c.M);
  check(cudaMalloc(&c.da,c.A.size()*8));check(cudaMalloc(&c.dr,c.R.size()*8));check(cudaMalloc(&c.ds,c.AS.size()*8));check(cudaMalloc(&c.dt,c.RS.size()*8));check(cudaMemcpy(c.da,c.A.data(),c.A.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(c.dr,c.R.data(),c.R.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(c.ds,c.AS.data(),c.AS.size()*8,cudaMemcpyHostToDevice));check(cudaMemcpy(c.dt,c.RS.data(),c.RS.size()*8,cudaMemcpyHostToDevice));for(auto&dp:c.dp)check(cudaMalloc(&dp,c.A.size()*8));for(auto*x:{&c.A,&c.R,&c.AS,&c.RS})prefix.insert(prefix.end(),x->begin(),x->end());prefix.insert(prefix.end(),3*c.A.size(),0);for(double v:{double(c.n),21.,double(c.M),3.,192.,96.,32256.})prefix.push_back(v);cases.push_back(std::move(c));
 }
 auto run=[&](Case&c,std::vector<double>&out,std::vector<double>&times,bool&ok,double&error){for(int variant=0;variant<3;++variant){check(cudaMemset(c.dp[variant],0,c.A.size()*8));check(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<3;++rep){if(variant==0)compute_probs<<<(c.n+191)/192,192>>>(c.da,c.dr,c.dp[variant],c.n,21,c.M);else if(variant==1)compute_probs_unitStrides<<<(c.n+191)/192,192>>>(c.ds,c.dt,c.dp[variant],c.n,21,c.M);else compute_probs_unitStrides_sharedMem<<<(c.n+95)/96,96,32256>>>(c.ds,c.dt,c.dp[variant],c.n,21,c.M);}check(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/3);std::vector<double>raw(c.A.size()),got(c.A.size());check(cudaMemcpy(raw.data(),c.dp[variant],raw.size()*8,cudaMemcpyDeviceToHost));for(int i=0;i<c.n;++i){double sum=0;for(int k=0;k<21;++k){size_t index=i*21+k;got[index]=variant==0?raw[index]:raw[k*c.n+i];error=std::max(error,fabs(got[index]-c.want[index]));ok=ok&&(llround(got[index]*c.M)==c.counts[index]);sum+=got[index];}ok=ok&&fabs(sum-1)<=1e-12;}out.insert(out.end(),got.begin(),got.end());}};
 for(auto&c:cases){std::vector<double>out,times;bool ok=true;double error=0;run(c,out,times,ok,error);if(!ok||error>.001)return 3;}
 for(int trial=1;trial<=5;++trial){auto out=prefix;std::vector<double>times;bool ok=true;double error=0;for(auto&c:cases)run(c,out,times,ok,error);ok=ok&&error<=.001;double elapsed=0;for(double x:times)elapsed+=x;std::string name="trial_"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();std::cout<<std::setprecision(17)<<"STUDY_TRIAL {\"trial\":"<<trial<<",\"time_seconds\":"<<elapsed<<",\"component_times_seconds\":[";for(size_t i=0;i<times.size();++i)std::cout<<(i?",":"")<<times[i];std::cout<<"],\"correct\":"<<(ok?"true":"false")<<",\"checked_values\":"<<out.size()<<",\"max_abs_error\":"<<error<<",\"timing_region\":\"sum_six_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\""<<name<<"\",\"dtype\":\"float64\",\"count\":"<<out.size()<<",\"byte_order\":\"little\"}}"<<std::endl;if(!ok)return 2;}
 for(auto&c:cases){cudaFree(c.da);cudaFree(c.dr);cudaFree(c.ds);cudaFree(c.dt);for(auto dp:c.dp)cudaFree(dp);}}
