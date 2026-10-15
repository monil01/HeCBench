// HPL DMem vendor shim. Original HPL CPU source/license: ../hpl-cuda/.
// This product includes software developed at the University of Tennessee,
// Knoxville, Innovative Computing Laboratory. See LICENSE.md.
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cblas.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
struct Shape {int m,n,k,lda,ldb,ldc;};
static int vendor(void* stream,unsigned,void*const* args){
 int op=hecbench_vendor_arg<int>(args,1);auto s=hecbench_vendor_arg<Shape>(args,2);
 double alpha=hecbench_vendor_arg<double>(args,3),beta=hecbench_vendor_arg<double>(args,4);
 auto a=hecbench_vendor_arg<const double*>(args,5);
 auto b=hecbench_vendor_arg<double*>(args,6);
 hipblasHandle_t h;if(hipblasCreate(&h)!=HIPBLAS_STATUS_SUCCESS)return 1;
 if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS)return 2;
 hipblasStatus_t status;
 if(op==0){auto c=hecbench_vendor_arg<double*>(args,7);status=hipblasDgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,s.m,s.n,s.k,&alpha,a,s.lda,b,s.ldb,&beta,c,s.ldc);}
 else status=hipblasDtrsm(h,HIPBLAS_SIDE_LEFT,HIPBLAS_FILL_MODE_LOWER,HIPBLAS_OP_N,HIPBLAS_DIAG_UNIT,s.m,s.n,&alpha,a,s.lda,b,s.ldb);
 auto sync=hipStreamSynchronize(static_cast<hipStream_t>(stream));auto destroy=hipblasDestroy(h);
 return status==HIPBLAS_STATUS_SUCCESS&&sync==hipSuccess&&destroy==HIPBLAS_STATUS_SUCCESS?0:3;
}
static void require(int status){if(status!=0){fprintf(stderr,"HPL IRIS operation failed %d\n",status);std::abort();}}
extern "C" void dpcpp_dgemm(int,int,int,int m,int n,int k,double alpha,const double*a,int lda,const double*b,int ldb,double beta,double*c,int ldc){
 if(m==0||n==0||k==0)return;
 // These CPU branches and thresholds are part of the original benchmark.
 if(n<64||m<128||k<128){cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);return;}
 iris::DMem da(const_cast<double*>(a),size_t(k)*lda*8),db(const_cast<double*>(b),size_t(n)*ldb*8),dc(c,size_t(n)*ldc*8);
 int op=0;Shape shape{m,n,k,lda,ldb,ldc};hecbench_vendor_callback cb=vendor;
 void* p[]={&cb,&op,&shape,&alpha,&beta,&da,&db,&dc};int pi[]={sizeof(cb),sizeof(op),sizeof(shape),sizeof(alpha),sizeof(beta),iris_r,iris_r,iris_rw};
 size_t g[]={1},l[]={1};iris::Task task;require(task.kernel("hecbench_vendor",1,nullptr,g,l,8,p,pi));require(task.flush_out(dc));require(task.submit(0,nullptr,1));
}
extern "C" void dpcpp_dtrsm(int,int,int,int,int,int m,int n,double alpha,const double*a,int lda,double*b,int ldb){
 if(m==0||n==0)return;
 if(m<512||n<2*m){cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit,m,n,alpha,a,lda,b,ldb);return;}
 iris::DMem da(const_cast<double*>(a),size_t(m)*lda*8),db(b,size_t(n)*ldb*8);
 int op=1;double beta=0;Shape shape{m,n,0,lda,ldb,0};hecbench_vendor_callback cb=vendor;
 void* p[]={&cb,&op,&shape,&alpha,&beta,&da,&db};int pi[]={sizeof(cb),sizeof(op),sizeof(shape),sizeof(alpha),sizeof(beta),iris_r,iris_rw};
 size_t g[]={1},l[]={1};iris::Task task;require(task.kernel("hecbench_vendor",1,nullptr,g,l,7,p,pi));require(task.flush_out(db));require(task.submit(0,nullptr,1));
}
extern "C" int hpl_driver_main(int,char**);
extern "C" void hecbench_report_vendor_calls();
extern "C" int __real_MPI_Finalize();
static iris::Platform* active_platform=nullptr;
static bool finalized=false;
static int finish(){
 if(finalized)return MPI_SUCCESS;finalized=true;
 hecbench_report_vendor_calls();bool error=iris_error_count()!=0;
 require(active_platform->finalize());require(__real_MPI_Finalize());
 if(error)std::abort();return MPI_SUCCESS;
}
extern "C" int __wrap_MPI_Init(int*,char***){return MPI_SUCCESS;}
extern "C" int __wrap_MPI_Finalize(){return finish();}
int main(int argc,char**argv){
 int provided;require(MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided));if(provided<MPI_THREAD_FUNNELED)std::abort();
 setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);iris::Platform platform;active_platform=&platform;require(platform.init(&argc,&argv,1));
 // The original driver calls exit(0), so finalize at its MPI_Finalize hook.
 // Also handle a future driver variant that returns normally.
 int status=hpl_driver_main(argc,argv);finish();return status;
}
