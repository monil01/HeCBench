#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#define main kmc_driver_main
#include "driver.c"
#undef main
static int gemv(void* stream,unsigned,void*const* args) {
 auto a=hecbench_vendor_arg<const float*>(args,1),x=hecbench_vendor_arg<const float*>(args,2);
 auto y=hecbench_vendor_arg<float*>(args,3);
 int rows=hecbench_vendor_arg<int>(args,4),features=hecbench_vendor_arg<int>(args,5);
 hipblasHandle_t h;if(hipblasCreate(&h)!=HIPBLAS_STATUS_SUCCESS)return 1;
 if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS)return 2;
 float one=1,zero=0;auto status=hipblasSgemv(h,HIPBLAS_OP_N,rows,features,&one,a,rows,x,1,&zero,y,1);
 auto sync=hipStreamSynchronize(static_cast<hipStream_t>(stream));auto destroy=hipblasDestroy(h);
 return status==HIPBLAS_STATUS_SUCCESS&&sync==hipSuccess&&destroy==HIPBLAS_STATUS_SUCCESS?0:3;
}
void cal_km(struct svm_problem* pecm) {
 int features=prob.x[0].dim,rows=prob.l;
 std::vector<float> a(size_t(rows)*features),trans(a.size()),x(features),dot(rows);
 std::vector<double> squared(rows,0);
 float gamma=param.gamma;double g=double(gamma);
 for(int r=0;r<rows;++r)for(int c=0;c<features;++c){float v=float(prob.x[r].values[c]);a[size_t(r)*features+c]=v;trans[size_t(c)*rows+r]=v;squared[r]+=pow(v,float(2));}
 auto offload=std::chrono::steady_clock::now();
 iris::DMem da(trans.data(),trans.size()*4),dx(x.data(),x.size()*4),dy(dot.data(),dot.size()*4);
 double total=0,worst=0;size_t errors=0;
 for(int r=0;r<rows;++r){
  auto start=std::chrono::steady_clock::now();
  std::memcpy(x.data(),a.data()+size_t(r)*features,x.size()*4);dx.update(x.data());
  hecbench_vendor_callback cb=gemv;void* p[]={&cb,&da,&dx,&dy,&rows,&features};
  int pi[]={sizeof(cb),iris_r,iris_r,iris_w,sizeof(rows),sizeof(features)};
  size_t gws[]={1},lws[]={1};iris::Task task;
  task.kernel("hecbench_vendor",1,nullptr,gws,lws,6,p,pi);task.flush_out(dy);
  if(task.submit(iris_gpu,nullptr,1))std::abort();
  total+=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
  pecm->x[r].values[0]=r+1;
  for(int c=0;c<rows;++c){
   double value=exp(-g*(squared[r]+squared[c]-2.0*double(dot[c])));
   pecm->x[r].values[c+1]=value;
   if(std::getenv("KMC_VALIDATE")){
    long double refdot=0;for(int k=0;k<features;++k)refdot+=(long double)a[size_t(r)*features+k]*a[size_t(c)*features+k];
    double ref=exp(-g*(squared[r]+squared[c]-2.0*double(refdot)));
    double delta=std::abs(value-ref);worst=std::max(worst,delta);
    if(!std::isfinite(value)||delta>1e-5+1e-4*std::abs(ref))++errors;
   }
   if(std::getenv("KMC_DUMP"))printf("MATRIX %d %d %.17g\n",r,c,value);
  }
 }
 printf("Average kernel matrix offload time: %lf (us)\n",total*1e-3/rows);
 printf("Total kernel matrix execution time: %lf (us)\n",std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-offload).count());
 if(std::getenv("KMC_VALIDATE"))printf("Independent full RBF matrix %zu values max_abs %.12g %s\n",size_t(rows)*rows,worst,errors?"FAIL":"PASS");
 if(errors)std::abort();
}
int main(int argc,char**argv){
 setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);iris::Platform platform;platform.init(&argc,&argv,1);
 // Original LIBSVM starts with the C runtime's default seed 1; IRIS init may
 // touch libc's global generator, so restore that original stream explicitly.
 srand(1);int status=kmc_driver_main(argc,argv);status|=iris_error_count()!=0;platform.finalize();return status;
}
