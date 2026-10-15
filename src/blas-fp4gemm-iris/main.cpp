#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "format.h"
static int gemm(void*stream,unsigned,void*const*a){
 auto h=hecbench_vendor_arg<hipblasHandle_t>(a,1);int M=hecbench_vendor_arg<int>(a,2),N=hecbench_vendor_arg<int>(a,3),K=hecbench_vendor_arg<int>(a,4);
 auto A=hecbench_vendor_arg<const float*>(a,5),B=hecbench_vendor_arg<const float*>(a,6);auto D=hecbench_vendor_arg<float*>(a,7);
 float alpha=1,beta=0;if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS)return 1;
 if(hipblasSgemm(h,HIPBLAS_OP_T,HIPBLAS_OP_N,M,N,K,&alpha,A,K,B,K,&beta,D,M)!=HIPBLAS_STATUS_SUCCESS)return 2;
 return int(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
}
static uint16_t bf(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float unbf(uint16_t u){uint32_t b=uint32_t(u)<<16;float v;memcpy(&v,&b,4);return v;}
// Independent host decoder: a lookup table generated from exact powers of two.
static double decode_code(unsigned u){unsigned mag=u&((1<<(FORMAT_BITS-1))-1),e=mag>>MANTISSA_BITS,m=mag% (1<<MANTISSA_BITS);double v=std::ldexp(double(m),1-EXPONENT_BIAS-MANTISSA_BITS);if(e)v=std::ldexp(double((1<<MANTISSA_BITS)+m),int(e)-EXPONENT_BIAS-MANTISSA_BITS);return (u>>(FORMAT_BITS-1))?-v:v;}
static double decode_scale(unsigned u){if(SCALE_E8M0)return std::ldexp(1.,int(u)-127);return (u>>3)?std::ldexp(double(8+(u&7)),int(u>>3)-10):std::ldexp(double(u&7),-9);}
static void launch(const char*name,size_t count,std::vector<void*>args,std::vector<int>info,iris::DMem*out=nullptr){size_t g[]={((count+255)/256)*256},l[]={256};iris::Task t;t.kernel(name,1,nullptr,g,l,args.size(),args.data(),info.data());if(out)t.flush_out(*out);t.submit(0,nullptr,1);}
static bool run(int M,int N,int K,int repeat,hipblasHandle_t handle,bool patterned){
 size_t na=size_t(M)*K,nb=size_t(N)*K,nd=size_t(M)*N;
 std::vector<uint8_t>A(na*FORMAT_BITS/8,0),B(nb*FORMAT_BITS/8,0),sa(na/SCALE_BLOCK,SCALE_FILL),sb(nb/SCALE_BLOCK,SCALE_FILL);
 std::vector<float>Af(na),Bf(nb),Df(nd);std::vector<uint16_t>D(nd);
 std::vector<double>lut(1u<<FORMAT_BITS);for(unsigned i=0;i<lut.size();++i)lut[i]=decode_code(i);
 auto fill=[&](auto&packed,size_t n,unsigned salt){for(size_t i=0;i<n;++i){unsigned c=patterned?unsigned((i*13+i/K*7+salt)%lut.size()):VALUE_FILL;size_t bit=i*FORMAT_BITS;packed[bit/8]|=uint8_t(c<<(bit%8));if(bit%8+FORMAT_BITS>8)packed[bit/8+1]|=uint8_t(c>>(8-bit%8));}};
 fill(A,na,3);fill(B,nb,5);
 if(patterned){for(size_t i=0;i<sa.size();++i)sa[i]=SCALE_E8M0?125+i%5:0x28+(i%5)*8;for(size_t i=0;i<sb.size();++i)sb[i]=SCALE_E8M0?125+(i*3)%5:0x28+((i*3)%5)*8;}
 iris::DMem ma(A.data(),A.size()),mb(B.data(),B.size()),msa(sa.data(),sa.size()),msb(sb.data(),sb.size()),maf(Af.data(),na*4),mbf(Bf.data(),nb*4),mdf(Df.data(),nd*4),md(D.data(),nd*2);
 launch("decode",na,{&ma,&msa,&maf,&M,&K},{iris_r,iris_r,iris_w,sizeof(M),sizeof(K)});
 launch("decode",nb,{&mb,&msb,&mbf,&N,&K},{iris_r,iris_r,iris_w,sizeof(N),sizeof(K)});
 auto invoke=[&]{hecbench_vendor_callback callback=gemm;launch("hecbench_vendor",1,{&callback,&handle,&M,&N,&K,&maf,&mbf,&mdf},{sizeof(callback),sizeof(handle),sizeof(M),sizeof(N),sizeof(K),iris_r,iris_r,iris_w});launch("bf16",nd,{&mdf,&md,&nd},{iris_r,iris_w,sizeof(nd)},&md);};
 invoke();auto start=std::chrono::steady_clock::now();for(int i=0;i<repeat;++i)invoke();double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeat;
 bool ok=true;double maxerr=0;size_t exact=0;
 auto get=[&](const auto&packed,size_t i){size_t bit=i*FORMAT_BITS;unsigned c=packed[bit/8]>>(bit%8);if(bit%8+FORMAT_BITS>8)c|=unsigned(packed[bit/8+1])<<(8-bit%8);return lut[c&(lut.size()-1)];};
 for(int j=0;j<N;++j)for(int i=0;i<M;++i){double expected=K;if(patterned){expected=0;for(int k=0;k<K;++k)expected+=get(A,size_t(i)*K+k)*decode_scale(sa[size_t(i)*(K/SCALE_BLOCK)+k/SCALE_BLOCK])*get(B,size_t(j)*K+k)*decode_scale(sb[size_t(j)*(K/SCALE_BLOCK)+k/SCALE_BLOCK]);}float got=unbf(D[size_t(j)*M+i]);double e=std::abs(got-expected);maxerr=std::max(maxerr,e);ok=ok&&std::isfinite(got)&&e<=1e-2*(std::abs(expected)+1);exact+=D[size_t(j)*M+i]==bf(float(expected));}
 printf("%s: %s shape=(%d,%d,%d) all_outputs=%zu BF16_exact=%zu max_abs=%.9g average_FP32_SGEMM_BF16_us=%.3f patterned=%d\n",ok?"PASS":"FAIL",FORMAT_NAME,M,N,K,nd,exact,maxerr,us,patterned);
 const char*dump=getenv("HECBENCH_IRIS_DUMP");if(dump){FILE*f=fopen(dump,"wb");if(!f||fwrite(D.data(),2,D.size(),f)!=D.size())throw std::runtime_error("dump failed");fclose(f);}return ok;
}
int main(int argc,char**argv){if(argc!=2||atoi(argv[1])<=0){printf("Usage: %s <repeat>\n",argv[0]);return 1;}int repeat=atoi(argv[1]);
 int shapes[6][3]={{16384,8192,1280},{16384,1024,8192},{16384,8192,7168},{16384,3584,8192},{8192,8192,8192},{16384,16384,16384}};int count=6;
 if(const char*s=getenv("HECBENCH_IRIS_SHAPE")){if(sscanf(s,"%d,%d,%d",&shapes[0][0],&shapes[0][1],&shapes[0][2])!=3)return 2;count=1;}
 for(int i=0;i<count;++i)if(shapes[i][0]<=0||shapes[i][1]<=0||shapes[i][2]<=0||shapes[i][2]%128)return 2;
 setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",1);iris::Platform platform;platform.init(&argc,&argv,1);hipblasHandle_t handle=nullptr;bool ok=hipblasCreate(&handle)==HIPBLAS_STATUS_SUCCESS;
 try{if(ok)for(int i=0;i<count;++i)ok=run(shapes[i][0],shapes[i][1],shapes[i][2],repeat,handle,getenv("HECBENCH_IRIS_PATTERNED")!=nullptr)&&ok;}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());ok=false;}
 if(handle)hipblasDestroy(handle);ok=ok&&iris_error_count()==0;platform.finalize();return ok?0:1;
}
