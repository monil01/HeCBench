#include <iris/iris.hpp>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstdint>
template<class T> T value(int i) { return T(float(i%255-127)/127); }
template<> double value<double>(int i) { return double(i%255-127)/127; }
template<> int8_t value<int8_t>(int i) { return int8_t((float(i%255-127)/127)*127); }
template<class T,class S> int run(const char* name,int m,int n,int k,int repeat,int mode) {
 std::vector<T> a(size_t(m)*k),b(size_t(k)*n);
 std::vector<S> c(size_t(m)*n);
 for(size_t i=0;i<a.size();++i)a[i]=value<T>(i);
 for(size_t i=0;i<b.size();++i)b[i]=value<T>(i);
 iris::DMem da(a.data(),a.size()*sizeof(T)),db(b.data(),b.size()*sizeof(T)),dc(c.data(),c.size()*sizeof(S));
 size_t g[]={((c.size()+255)/256)*256},l[]={256};
 auto start=std::chrono::steady_clock::now();
 for(int r=0;r<repeat;++r) {
  iris::Task task;void* p[]={&da,&db,&dc,&m,&n,&k};
  int pi[]={iris_r,iris_r,iris_w,sizeof(m),sizeof(n),sizeof(k)};
  task.kernel(name,1,nullptr,g,l,6,p,pi); if(r==repeat-1)task.flush_out(dc);
  if(task.submit(iris_gpu,nullptr,1)!=0)return 1;
 }
 double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 double worst=0;size_t errors=0;
 for(int row=0;row<m;++row)for(int col=0;col<n;++col) {
  long double sum=0;
  for(int q=0;q<k;++q)sum+=(long double)double(a[size_t(row)*k+q])*double(b[size_t(q)*n+col]);
  double ref=double(S(sum)),got=double(c[size_t(row)*n+col]);
  double tol=std::is_same<S,double>::value?1e-10:(std::is_integral<S>::value?0:0.008+0.008*std::abs(ref));
  worst=std::max(worst,std::abs(got-ref));
  if(!std::isfinite(got)||std::abs(got-ref)>tol)++errors;
  if(std::getenv("IRIS_GEMM_DUMP"))printf("RAW %d %d %.17g\n",mode,row*n+col,got);
 }
 printf("%s mode %d full-output %zu values max_abs_error %.12g seconds %.9f %s\n",name,mode,c.size(),worst,sec,errors?"FAIL":"PASS");
 return errors!=0;
}
int main(int argc,char**argv) {
 if(argc!=5){printf("Usage: %s <M> <N> <K> <iterations>\n",argv[0]);return 1;}
 int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),repeat=atoi(argv[4]);
 if(m<=0||n<=0||k<=0||repeat<=0)return 1;
 iris::Platform platform;platform.init(&argc,&argv,1);int errors=0,mode=0;
 errors+=run<double,double>("gemm_double",m,n,k,repeat,mode++);
 for(int i=0;i<4;++i)errors+=run<float,float>("gemm_float",m,n,k,repeat,mode++);
 errors+=run<__half,__half>("gemm_half",m,n,k,repeat,mode++);
 errors+=run<__half,__half>("gemm_half",m,n,k,repeat,mode++);
 errors+=run<hip_bfloat16,hip_bfloat16>("gemm_bfloat",m,n,k,repeat,mode++);
 errors+=run<int8_t,int32_t>("gemm_int",m,n,k,repeat,mode++);
 platform.finalize();return errors?1:0;
}
