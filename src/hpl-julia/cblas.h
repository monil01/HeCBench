#pragma once
#define CblasColMajor 102
#define CblasNoTrans 111
#define CblasLeft 141
#define CblasLower 122
#define CblasUnit 132
#ifdef __cplusplus
extern "C" {
#endif
int cblas_idamax(int n,const double*x,int incx);
void cblas_dswap(int n,double*x,int incx,double*y,int incy);
void cblas_dcopy(int n,const double*x,int incx,double*y,int incy);
void cblas_daxpy(int n,double a,const double*x,int incx,double*y,int incy);
void cblas_dscal(int n,double a,double*x,int incx);
void cblas_dgemv(int o,int t,int m,int n,double a,const double*A,int lda,const double*x,int ix,double b,double*y,int iy);
void cblas_dger(int o,int m,int n,double a,const double*x,int ix,const double*y,int iy,double*A,int lda);
void cblas_dtrsv(int o,int u,int t,int d,int n,const double*A,int lda,double*x,int ix);
void cblas_dgemm(int o,int ta,int tb,int m,int n,int k,double a,const double*A,int lda,const double*B,int ldb,double b,double*C,int ldc);
void cblas_dtrsm(int o,int s,int u,int t,int d,int m,int n,double a,const double*A,int lda,double*B,int ldb);
#ifdef __cplusplus
}
#endif
