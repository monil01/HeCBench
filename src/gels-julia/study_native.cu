#define main original_main
#include "../gels-cuda/main.cu"
#undef main
#include "study_fixture.h"
#include <vector>
#include <string>
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));
exit(2);
}}
template<typename T>struct Case{T*a,*b;
T**ap,**bp;
std::vector<T>ah,bh;
Case(){using R=decltype(std::real(T{}));
for(double x:INPUT_A){if constexpr(std::is_same_v<R,T>)ah.push_back(T(x));

else ah.push_back(T(0,R(x)));
}for(double x:INPUT_B){if constexpr(std::is_same_v<R,T>)bh.push_back(T(x));

else bh.push_back(T(0,R(x)));
}ck(cudaMalloc(&a,ah.size()*sizeof(T)));
ck(cudaMalloc(&b,bh.size()*sizeof(T)));
ck(cudaMalloc(&ap,2*sizeof(T*)));
ck(cudaMalloc(&bp,2*sizeof(T*)));
T*pa[]={a,a+25};
T*pb[]={b,b+5};
ck(cudaMemcpy(ap,pa,sizeof(pa),cudaMemcpyHostToDevice));
ck(cudaMemcpy(bp,pb,sizeof(pb),cudaMemcpyHostToDevice));
}void prefix(std::vector<double>&out,int code){for(auto&data:{ah,bh})for(auto x:data){out.push_back(std::real(x));
out.push_back(std::imag(x));
}out.insert(out.end(),{double(code),5,5,1,2,double(sizeof(T))});
}double run(cublasHandle_t h,std::vector<double>&out,bool&ok,double&error){ck(cudaMemcpy(a,ah.data(),ah.size()*sizeof(T),cudaMemcpyHostToDevice));
ck(cudaMemcpy(b,bh.data(),bh.size()*sizeof(T),cudaMemcpyHostToDevice));
ck(cudaDeviceSynchronize());
int info=0;
cublasStatus_t status;
auto start=std::chrono::steady_clock::now();
if constexpr(std::is_same_v<T,float>)status=cublasSgelsBatched(h,CUBLAS_OP_N,5,5,1,ap,5,bp,5,&info,nullptr,2);

else if constexpr(std::is_same_v<T,double>)status=cublasDgelsBatched(h,CUBLAS_OP_N,5,5,1,ap,5,bp,5,&info,nullptr,2);

else if constexpr(std::is_same_v<T,std::complex<float>>)status=cublasCgelsBatched(h,CUBLAS_OP_N,5,5,1,reinterpret_cast<cuComplex*const*>(ap),5,reinterpret_cast<cuComplex*const*>(bp),5,&info,nullptr,2);

else status=cublasZgelsBatched(h,CUBLAS_OP_N,5,5,1,reinterpret_cast<cuDoubleComplex*const*>(ap),5,reinterpret_cast<cuDoubleComplex*const*>(bp),5,&info,nullptr,2);
ck(cudaDeviceSynchronize());
double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
out.push_back(status);
out.push_back(info);
std::vector<T>got(10);
ck(cudaMemcpy(got.data(),b,10*sizeof(T),cudaMemcpyDeviceToHost));
using R=decltype(std::real(T{}));
double bound=(std::is_same_v<R, float>) ? 1e-6 : 1e-8;
ok&=status==0&&info==0;
for(auto x:got){double e=std::abs(x-T(1));
error=std::max(error,e);
ok&=std::isfinite(e)&&e<bound;
out.push_back(std::real(x));
out.push_back(std::imag(x));
}return elapsed;
}void free(){cudaFree(a);
cudaFree(b);
cudaFree(ap);
cudaFree(bp);
}};

int main(){cublasHandle_t h;
if(cublasCreate(&h)!=0)return 2;
int version=0;
cublasGetVersion(h,&version);
printf("STUDY_LIBRARY version=%d\n",version);
Case<float>s;
Case<std::complex<float>>c;
Case<double>d;
Case<std::complex<double>>z;
std::vector<double>prefix;
s.prefix(prefix,0);
c.prefix(prefix,1);
d.prefix(prefix,2);
z.prefix(prefix,3);
{std::vector<double>unused;
bool ok=true;
double error=0;
s.run(h,unused,ok,error);
c.run(h,unused,ok,error);
d.run(h,unused,ok,error);
z.run(h,unused,ok,error);
}for(int trial=1;
trial<=5;
++trial){auto out=prefix;
bool ok=true;
double error=0;
double elapsed=0; elapsed+=s.run(h,out,ok,error); elapsed+=c.run(h,out,ok,error); elapsed+=d.run(h,out,ok,error); elapsed+=z.run(h,out,ok,error);
std::string path="trial_"+std::to_string(trial)+".bin";
FILE*f=fopen(path.c_str(),"wb");
fwrite(out.data(),8,out.size(),f);
fclose(f);
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum_four_cublas_gels_batched_calls\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",out.size(),error,path.c_str(),out.size());
if(!ok)return 3;
}s.free();
c.free();
d.free();
z.free();
cublasDestroy(h);
return 0;
}
