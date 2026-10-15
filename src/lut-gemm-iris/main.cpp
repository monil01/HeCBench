#include <iris/iris.hpp>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "vendor.cpp"
bool rand_bool(){return rand()>(RAND_MAX/2);}
double rand_fp64(double max=1.0){double sign[]={-1.,1.};return sign[rand_bool()]*rand()/RAND_MAX*rand()/RAND_MAX*max;}
float rand_fp32(float max=1.0){return rand_fp64()*max;}
int run(int K,int repeat,int bits){
 int N=4*K,M=1,groups=K/128;
 std::vector<__half> input(K),alpha(size_t(groups)*bits*N),bias(size_t(groups)*N),weight(size_t(K)*N),out(N),ref(N);
 std::vector<uint32_t> packed(size_t(K/32)*bits*N);
 srand(123);for(auto&x:input)x=__float2half(rand_fp32());
 srand(123);
 for(int n=0;n<N;++n)for(int b=0;b<bits;++b)for(int k=0;k<K;k+=32){uint32_t w=0;for(int t=0;t<32;++t)if(rand_bool())w|=uint32_t(1)<<t;packed[size_t(k/32)*bits*N+b*N+n]=w;}
 std::vector<float> af(alpha.size()),bf(bias.size());
 srand(123);for(int g=0;g<groups;++g)for(int n=0;n<N;++n){bf[size_t(g)*N+n]=rand_fp32()/(1<<bits);for(int b=0;b<bits;++b)af[size_t(g)*bits*N+b*N+n]=rand_fp32()/(1<<b);}
 for(size_t i=0;i<af.size();++i)alpha[i]=__float2half(af[i]);for(size_t i=0;i<bf.size();++i)bias[i]=__float2half(bf[i]);
 for(int n=0;n<N;++n)for(int k=0;k<K;++k){
  float w=bf[size_t(k/128)*N+n];
  for(int b=0;b<bits;++b){uint32_t pw=packed[size_t(k/32)*bits*N+b*N+n];w+=af[size_t(k/128)*bits*N+b*N+n]*(((pw>>(k%32))&1)?1.f:-1.f);}
  weight[size_t(n)*K+k]=__float2half(w);
 }
 iris::DMem di(input.data(),input.size()*2),da(alpha.data(),alpha.size()*2),db(bias.data(),bias.size()*2),dw(packed.data(),packed.size()*4),df(weight.data(),weight.size()*2),dout(out.data(),out.size()*2),dref(ref.data(),ref.size()*2);
 {
 hecbench_vendor_callback callback=lut_reference;
 void*p[]={&callback,&di,&df,&dref,&M,&N,&K};int pi[]={sizeof(callback),iris_r,iris_r,iris_w,sizeof(M),sizeof(N),sizeof(K)};size_t g[]={1},l[]={1};
 iris::Task t;t.kernel("hecbench_vendor",1,nullptr,g,l,7,p,pi);t.flush_out(dref);if(t.submit(iris_gpu,nullptr,1))return 1;
 }
 int tile=2048,group=128;
 auto lut=[&](){
 size_t g0[]={size_t((N+255)/256)*256},l0[]={256};void*p0[]={&dout,&N};int pi0[]={iris_w,sizeof(N)};
 iris::Task zero;zero.kernel("clear_output",1,nullptr,g0,l0,2,p0,pi0);if(zero.submit(iris_gpu,nullptr,1))return 1;
 void*p[]={&dw,&da,&db,&di,&dout,&N,&K,&bits,&tile,&group};
 int pi[]={iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(N),sizeof(K),sizeof(bits),sizeof(tile),sizeof(group)};
 size_t g[]={size_t((N+tile-1)/tile)*256,size_t(K/32)},l[]={256,1};iris::Task t;t.kernel("lut_bias",2,nullptr,g,l,10,p,pi);t.flush_out(dout);return t.submit(iris_gpu,nullptr,1);
 };
 if(lut())return 1;
 double err=0;for(int n=0;n<N;++n)err+=std::abs(float(ref[n])-float(out[n]));
 printf("LUT-GEMM INT%d mean Error: %lf\n",bits,err/N);
 if(lut())return 1;auto start=std::chrono::steady_clock::now();for(int r=0;r<repeat;++r)if(lut())return 1;
 printf("Average latency %.9g ms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat);
 // Independent host simulation of the packed-byte lookup and FP16 arithmetic.
 // It does not produce or replace the GPU result; atomic tile accumulation
 // order on the GPU is allowed to differ from this ascending-tile reference.
 std::vector<__half> oracle(N,__float2half(0));
 std::vector<double> abs_tiles(N,0),bounds(N,0);
 for(int tile_k=0;tile_k<K;tile_k+=32){
  __half table[4][256];
  for(int y=0;y<4;++y){
   __half* inp=input.data()+tile_k+8*y;
   for(int x=0;x<64;++x){
    __half v=__float2half(2*((x>>0)&1)-1)*inp[0];
    for(int bit=1;bit<8;++bit)v=v+__float2half(2*((x>>bit)&1)-1)*inp[bit];
    table[y][x]=v;
   }
   for(int s=6;s<8;++s)for(int x=0;x<(1<<s);++x)table[y][(1<<s)+x]=table[y][x]+__float2half(2)*inp[s];
  }
  for(int n=0;n<N;++n){
   __half base=0;for(int y=0;y<4;++y)base=base+table[y][255];
   __half v=bias[size_t(tile_k/128)*N+n]*base;
   for(int b=0;b<bits;++b){
    uint32_t word=packed[size_t(tile_k/32)*bits*N+b*N+n];__half sum=0;
    for(int y=0;y<4;++y)sum=sum+table[y][(word>>(8*y))&255];
    v=v+alpha[size_t(tile_k/128)*bits*N+b*N+n]*sum;
   }
   oracle[n]=oracle[n]+v;abs_tiles[n]+=std::abs(float(v));
  }
 }
 // Two orderings of t FP16 additions differ by at most 2*gamma_t*sum|v|,
 // gamma_t=t*u/(1-t*u), u=2^-11 (round-to-nearest half unit roundoff).
 double tu=(K/32)*(1.0/2048),gamma=tu/(1-tu);
 int errors=0;double oracle_worst=0;
 for(int n=0;n<N;++n){bounds[n]=std::max(1e-3,2*gamma*abs_tiles[n]);double delta=std::abs(float(out[n])-float(oracle[n]));oracle_worst=std::max(oracle_worst,delta);if(delta>bounds[n])++errors;}
 printf("Independent packed-LUT oracle %d outputs max_abs %.9g %s\n",N,oracle_worst,errors?"FAIL":"PASS");
 for(int n=0;n<N;++n){if(!std::isfinite(float(out[n]))||!std::isfinite(float(ref[n])))++errors;if(std::getenv("IRIS_LUT_DUMP"))printf("VALUE %d %d %.9g %.9g %.9g %.17g\n",bits,n,float(out[n]),float(ref[n]),float(oracle[n]),bounds[n]);}
 printf("Full finite-output %d values %s\n",2*N,errors?"FAIL":"PASS");return errors?1:0;
}
int main(int argc,char**argv){
 if(argc!=3){printf("Usage: %s <test case> <repeat>\nCase 0: K = 1024\nCase 1: K = 4096\nDefault: K = 12288\n",argv[0]);return 1;}
 int option=atoi(argv[1]),repeat=atoi(argv[2]);if(repeat<1)return 1;int K=option==0?1024:option==1?4096:12288;
 setenv("IRIS_KERNEL_HOST2HIP","./libvendor.so",0);iris::Platform platform;platform.init(&argc,&argv,1);
 int error=0;for(int bits:{8,4,3})error|=run(K,repeat,bits);platform.finalize();return error;
}
