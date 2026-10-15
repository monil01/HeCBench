#include <iris/iris.hpp>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "mv.h"
static void check(int result){if(result!=IRIS_SUCCESS){fprintf(stderr,"IRIS error %d\n",result);exit(2);}}
long reference(int repeat,size_t n,const REAL* x,REAL* matrix,REAL* y){
 iris::DMem ma(matrix,n*n*sizeof(REAL)),mx(const_cast<REAL*>(x),n*sizeof(REAL)),my(y,n*sizeof(REAL));
 size_t local=256,global=(n+local-1)/local*local;auto start=std::chrono::steady_clock::now();
 for(int i=0;i<repeat;++i){iris::Task t;void* p[]={&n,&ma,&mx,&my};int modes[]={sizeof(n),iris_r,iris_r,iris_w};
 check(t.kernel("mv_dense",1,nullptr,&global,&local,4,p,modes));if(i==repeat-1)check(t.flush_out(my));check(t.submit(iris_gpu,nullptr,true));}
 return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
}
static long sparse(bool csr,int repeat,size_t n,const REAL* x,size_t nnz,REAL* matrix,REAL* y){
 std::vector<size_t> rows(csr?n+1:nnz),columns(nnz);std::vector<REAL> values(nnz);
 if(csr)init_csr(rows.data(),values.data(),columns.data(),matrix,n,nnz);else init_coo(rows.data(),values.data(),columns.data(),matrix,n,nnz);
 iris::DMem mr(rows.data(),rows.size()*sizeof(size_t)),mc(columns.data(),nnz*sizeof(size_t)),mv(values.data(),nnz*sizeof(REAL)),mx(const_cast<REAL*>(x),n*sizeof(REAL)),my(y,n*sizeof(REAL));
 size_t local=256,global=(n+local-1)/local*local;auto start=std::chrono::steady_clock::now();
 for(int i=0;i<repeat;++i){iris::Task t;void* p[]={&mr,&mc,&mv,&mx,&my,&n,&nnz};int modes[]={iris_r,iris_r,iris_r,iris_r,iris_w,sizeof(n),sizeof(nnz)};
 check(t.kernel(csr?"mv_csr":"mv_coo",1,nullptr,&global,&local,7,p,modes));if(i==repeat-1)check(t.flush_out(my));check(t.submit(iris_gpu,nullptr,true));}
 return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
}
long spmv_csr(int r,size_t n,const REAL* x,size_t nnz,REAL* matrix,REAL* y){return sparse(true,r,n,x,nnz,matrix,y);}
long spmv_coo(int r,size_t n,const REAL* x,size_t nnz,REAL* matrix,REAL* y){return sparse(false,r,n,x,nnz,matrix,y);}
