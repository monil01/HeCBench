#include <hipblas/hipblas.h>
#include <atomic>
#include <cstdio>
static std::atomic<unsigned long long> gemms{0},trsms{0};
extern "C" {
hipblasStatus_t __real_hipblasDgemm(hipblasHandle_t,hipblasOperation_t,hipblasOperation_t,int,int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
hipblasStatus_t __real_hipblasDtrsm(hipblasHandle_t,hipblasSideMode_t,hipblasFillMode_t,hipblasOperation_t,hipblasDiagType_t,int,int,const double*,const double*,int,double*,int);
hipblasStatus_t __wrap_hipblasDgemm(hipblasHandle_t h,hipblasOperation_t ta,hipblasOperation_t tb,int m,int n,int k,const double* alpha,const double*a,int lda,const double*b,int ldb,const double*beta,double*c,int ldc){
 ++gemms;return __real_hipblasDgemm(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
hipblasStatus_t __wrap_hipblasDtrsm(hipblasHandle_t h,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t trans,hipblasDiagType_t diag,int m,int n,const double*alpha,const double*a,int lda,double*b,int ldb){
 ++trsms;return __real_hipblasDtrsm(h,side,uplo,trans,diag,m,n,alpha,a,lda,b,ldb);
}
void hecbench_report_vendor_calls(){printf("GPU_VENDOR_CALLS dgemm=%llu dtrsm=%llu\n",gemms.load(),trsms.load());}
}
