
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cstring>
#define fp float
#define EQUATIONS 91
#define PARAMETERS 18
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel_fin.cu"
using Callback=double(*)(float,const float*,const float*,float*,float*);
static Callback evaluator=nullptr;
static std::vector<double> trace;
static int nonfinite_count=0;
static double kernel_seconds=0;
static double master(float t,float*input,float*params,float*out,float*com,float*,float*,float*,float*){
 double elapsed=evaluator(t,input,params,out,com);kernel_seconds+=elapsed;
 trace.push_back(t);trace.push_back((int)t);
 for(int i=0;i<91;i++)trace.push_back(input[i]);
 for(int i=0;i<91;i++)trace.push_back(out[i]);
 for(int i=0;i<3;i++)trace.push_back(com[i]);
 kernel_fin(input,0,46,61,76,params,out,com[0],com[1],com[2]);
 for(int i=0;i<91;i++){if(!std::isfinite(out[i])){nonfinite_count++;out[i]=.0001f;}trace.push_back(out[i]);}
 return elapsed*1e9;
}
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/embedded_fehlberg_7_8.cu"
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/solver.cu"
#undef min
#undef max
static std::vector<double> result;
extern "C" double study_solve(void*cb,const float*input,const float*params,int xmax){
 evaluator=(Callback)cb;trace.clear();nonfinite_count=0;kernel_seconds=0;
 std::vector<std::vector<float>> y(xmax+1,std::vector<float>(91));
 std::vector<float*> ptrs(xmax+1);for(int i=0;i<=xmax;i++)ptrs[i]=y[i].data();
 std::memcpy(ptrs[0],input,91*4);std::vector<float>x(xmax+1);float par[18],com[3]={};std::memcpy(par,params,18*4);
 int status=solver(ptrs.data(),x.data(),xmax,par,com,nullptr,nullptr,nullptr,nullptr);
 result.clear();result.push_back(status);result.push_back(nonfinite_count);result.push_back(trace.size()/278);
 for(auto v:x)result.push_back(v);
 for(auto &state:y)for(float v:state)result.push_back(v);
 result.insert(result.end(),trace.begin(),trace.end());
 return kernel_seconds;
}
extern "C" size_t study_count(){return result.size();}
extern "C" void study_copy(double*out){std::memcpy(out,result.data(),result.size()*8);}
