// LP64 HPL scalars widened explicitly to existing OpenBLAS ILP64 ABI.
extern long cblas_idamax64_(long n,const double*x,long incx);
int cblas_idamax(int n,const double*x,int incx){ return (int)cblas_idamax64_(n,x,incx);}
extern void cblas_dswap64_(long n,double*x,long incx,double*y,long incy);
void cblas_dswap(int n,double*x,int incx,double*y,int incy){ cblas_dswap64_(n,x,incx,y,incy);}
extern void cblas_dcopy64_(long n,const double*x,long incx,double*y,long incy);
void cblas_dcopy(int n,const double*x,int incx,double*y,int incy){ cblas_dcopy64_(n,x,incx,y,incy);}
extern void cblas_daxpy64_(long n,double a,const double*x,long incx,double*y,long incy);
void cblas_daxpy(int n,double a,const double*x,int incx,double*y,int incy){ cblas_daxpy64_(n,a,x,incx,y,incy);}
extern void cblas_dscal64_(long n,double a,double*x,long incx);
void cblas_dscal(int n,double a,double*x,int incx){ cblas_dscal64_(n,a,x,incx);}
extern void cblas_dgemv64_(int o,int t,long m,long n,double a,const double*A,long lda,const double*x,long ix,double b,double*y,long iy);
void cblas_dgemv(int o,int t,int m,int n,double a,const double*A,int lda,const double*x,int ix,double b,double*y,int iy){ cblas_dgemv64_(o,t,m,n,a,A,lda,x,ix,b,y,iy);}
extern void cblas_dger64_(int o,long m,long n,double a,const double*x,long ix,const double*y,long iy,double*A,long lda);
void cblas_dger(int o,int m,int n,double a,const double*x,int ix,const double*y,int iy,double*A,int lda){ cblas_dger64_(o,m,n,a,x,ix,y,iy,A,lda);}
extern void cblas_dtrsv64_(int o,int u,int t,int d,long n,const double*A,long lda,double*x,long ix);
void cblas_dtrsv(int o,int u,int t,int d,int n,const double*A,int lda,double*x,int ix){ cblas_dtrsv64_(o,u,t,d,n,A,lda,x,ix);}
extern void cblas_dgemm64_(int o,int ta,int tb,long m,long n,long k,double a,const double*A,long lda,const double*B,long ldb,double b,double*C,long ldc);
void cblas_dgemm(int o,int ta,int tb,int m,int n,int k,double a,const double*A,int lda,const double*B,int ldb,double b,double*C,int ldc){ cblas_dgemm64_(o,ta,tb,m,n,k,a,A,lda,B,ldb,b,C,ldc);}
extern void cblas_dtrsm64_(int o,int s,int u,int t,int d,long m,long n,double a,const double*A,long lda,double*B,long ldb);
void cblas_dtrsm(int o,int s,int u,int t,int d,int m,int n,double a,const double*A,int lda,double*B,int ldb){ cblas_dtrsm64_(o,s,u,t,d,m,n,a,A,lda,B,ldb);}
