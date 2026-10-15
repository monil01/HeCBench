#include "solver.hpp"
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
struct State{
 int m,nnz,maxIterations,verbose;double residual=0;
 size_t bytesMV=0,bytesL=0,bytesU=0;int bytesLU=0;
 hipblasHandle_t blas=nullptr;hipsparseHandle_t sparse=nullptr;
 hipsparseSpMatDescr_t a=nullptr,lower=nullptr,upper=nullptr;
 hipsparseMatDescr_t lu=nullptr;csrilu02Info_t info=nullptr;
 Vec b{},x{},r{},r0{},p{},pa{},s{},sa{},v{},t{},tmp{};
};
static int vendor(void* rawstream,unsigned,void* const* args){
 State* s=hecbench_vendor_arg<State*>(args,1);int phase=hecbench_vendor_arg<int>(args,2);
 int* rows=hecbench_vendor_arg<int*>(args,3);int* cols=hecbench_vendor_arg<int*>(args,4);
 double* av=hecbench_vendor_arg<double*>(args,5);double* mv=hecbench_vendor_arg<double*>(args,6);double* vectors=hecbench_vendor_arg<double*>(args,7);
 hipStream_t stream=reinterpret_cast<hipStream_t>(rawstream);
 Vec* vecs[]={&s->b,&s->x,&s->r,&s->r0,&s->p,&s->pa,&s->s,&s->sa,&s->v,&s->t,&s->tmp};
 for(int i=0;i<11;++i)vecs[i]->ptr=vectors+size_t(i)*s->m;
 const double one=1,zero=0;const double* zeros=vectors+size_t(11)*s->m;
 if(phase==0){
 CHECK_HIPBLAS(hipblasCreate(&s->blas));CHECK_HIPSPARSE(hipsparseCreate(&s->sparse));
 CHECK_HIPBLAS(hipblasSetStream(s->blas,stream));CHECK_HIPSPARSE(hipsparseSetStream(s->sparse,stream));
 CHECK_HIPBLAS(hipblasSetPointerMode(s->blas,HIPBLAS_POINTER_MODE_HOST));CHECK_HIPSPARSE(hipsparseSetPointerMode(s->sparse,HIPSPARSE_POINTER_MODE_HOST));
 for(Vec* v:vecs)CHECK_HIPSPARSE(hipsparseCreateDnVec(&v->vec,s->m,v->ptr,HIP_R_64F));
 CHECK_HIPSPARSE(hipsparseCreateCsr(&s->a,s->m,s->m,s->nnz,rows,cols,av,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_BASE_ZERO,HIP_R_64F));
 CHECK_HIPSPARSE(hipsparseCreateCsr(&s->lower,s->m,s->m,s->nnz,rows,cols,mv,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_BASE_ZERO,HIP_R_64F));
 CHECK_HIPSPARSE(hipsparseCreateCsr(&s->upper,s->m,s->m,s->nnz,rows,cols,mv,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_BASE_ZERO,HIP_R_64F));
 hipsparseFillMode_t lower=HIPSPARSE_FILL_MODE_LOWER,upper=HIPSPARSE_FILL_MODE_UPPER;
 hipsparseDiagType_t unit=HIPSPARSE_DIAG_TYPE_UNIT,nonunit=HIPSPARSE_DIAG_TYPE_NON_UNIT;
 CHECK_HIPSPARSE(hipsparseSpMatSetAttribute(s->lower,HIPSPARSE_SPMAT_FILL_MODE,&lower,sizeof(lower)));
 CHECK_HIPSPARSE(hipsparseSpMatSetAttribute(s->lower,HIPSPARSE_SPMAT_DIAG_TYPE,&unit,sizeof(unit)));
 CHECK_HIPSPARSE(hipsparseSpMatSetAttribute(s->upper,HIPSPARSE_SPMAT_FILL_MODE,&upper,sizeof(upper)));
 CHECK_HIPSPARSE(hipsparseSpMatSetAttribute(s->upper,HIPSPARSE_SPMAT_DIAG_TYPE,&nonunit,sizeof(nonunit)));
 const double alpha=.75;
 CHECK_HIPSPARSE(hipsparseSpMV_bufferSize(s->sparse,HIPSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s->a,s->x.vec,&zero,s->b.vec,HIP_R_64F,HIPSPARSE_SPMV_ALG_DEFAULT,&s->bytesMV));
 CHECK_HIPSPARSE(hipsparseCreateMatDescr(&s->lu));CHECK_HIPSPARSE(hipsparseSetMatType(s->lu,HIPSPARSE_MATRIX_TYPE_GENERAL));CHECK_HIPSPARSE(hipsparseSetMatIndexBase(s->lu,HIPSPARSE_INDEX_BASE_ZERO));CHECK_HIPSPARSE(hipsparseCreateCsrilu02Info(&s->info));
 CHECK_HIPSPARSE(hipsparseDcsrilu02_bufferSize(s->sparse,s->m,s->nnz,s->lu,mv,rows,cols,s->info,&s->bytesLU));
 hipsparseSpSVDescr_t dl,du;CHECK_HIPSPARSE(hipsparseSpSV_createDescr(&dl));CHECK_HIPSPARSE(hipsparseSpSV_createDescr(&du));
 CHECK_HIPSPARSE(hipsparseSpSV_bufferSize(s->sparse,HIPSPARSE_OPERATION_NON_TRANSPOSE,&one,s->lower,s->p.vec,s->tmp.vec,HIP_R_64F,HIPSPARSE_SPSV_ALG_DEFAULT,dl,&s->bytesL));
 CHECK_HIPSPARSE(hipsparseSpSV_bufferSize(s->sparse,HIPSPARSE_OPERATION_NON_TRANSPOSE,&one,s->upper,s->tmp.vec,s->pa.vec,HIP_R_64F,HIPSPARSE_SPSV_ALG_DEFAULT,du,&s->bytesU));
 CHECK_HIPSPARSE(hipsparseSpSV_destroyDescr(dl));CHECK_HIPSPARSE(hipsparseSpSV_destroyDescr(du));
 }else{
 void* bufMV=hecbench_vendor_arg<void*>(args,8);void* bufLU=hecbench_vendor_arg<void*>(args,9);void* bufL=hecbench_vendor_arg<void*>(args,10);void* bufU=hecbench_vendor_arg<void*>(args,11);
 // Native hipMalloc(size0) yields nullptr: a non-null dummy changes SpMV semantics.
 if(s->bytesMV==0)bufMV=nullptr;
 CHECK_HIPBLAS(hipblasSetStream(s->blas,stream));CHECK_HIPSPARSE(hipsparseSetStream(s->sparse,stream));
 CHECK_HIPBLAS(hipblasSetPointerMode(s->blas,HIPBLAS_POINTER_MODE_HOST));CHECK_HIPSPARSE(hipsparseSetPointerMode(s->sparse,HIPSPARSE_POINTER_MODE_HOST));
 for(Vec* v:vecs)CHECK_HIPSPARSE(hipsparseDnVecSetValues(v->vec,v->ptr));
 CHECK_HIPSPARSE(hipsparseCsrSetPointers(s->a,rows,cols,av));CHECK_HIPSPARSE(hipsparseCsrSetPointers(s->lower,rows,cols,mv));CHECK_HIPSPARSE(hipsparseCsrSetPointers(s->upper,rows,cols,mv));
 if(getenv("HECBENCH_IRIS_DIAGNOSTICS"))fprintf(stderr,"DMem workspace bytes=%zu/%d/%zu/%zu\n",s->bytesMV,s->bytesLU,s->bytesL,s->bytesU);
 const double alpha=.75;
 CHECK_HIPSPARSE(hipsparseSpMV(s->sparse,HIPSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s->a,s->x.vec,&zero,s->b.vec,HIP_R_64F,HIPSPARSE_SPMV_ALG_DEFAULT,bufMV));
 CHECK_HIPBLAS(hipblasDcopy(s->blas,s->m,zeros,1,s->x.ptr,1));
 CHECK_HIPSPARSE(hipsparseDcsrilu02_analysis(s->sparse,s->m,s->nnz,s->lu,mv,rows,cols,s->info,HIPSPARSE_SOLVE_POLICY_USE_LEVEL,bufLU));
 int pivot;CHECK_HIPSPARSE(hipsparseXcsrilu02_zeroPivot(s->sparse,s->info,&pivot));
 CHECK_HIPSPARSE(hipsparseDcsrilu02(s->sparse,s->m,s->nnz,s->lu,mv,rows,cols,s->info,HIPSPARSE_SOLVE_POLICY_USE_LEVEL,bufLU));
 CHECK_HIPSPARSE(hipsparseXcsrilu02_zeroPivot(s->sparse,s->info,&pivot));
 CHECK_HIPSPARSE(hipsparseDestroyCsrilu02Info(s->info));CHECK_HIPSPARSE(hipsparseDestroyMatDescr(s->lu));
 CHECK_HIP(hipStreamSynchronize(stream));auto start=std::chrono::steady_clock::now();
 s->residual=gpu_BiCGStab(s->verbose,s->blas,s->sparse,s->m,s->a,s->lower,s->upper,s->b,s->x,s->r0,s->r,s->p,s->pa,s->s,s->sa,s->v,s->t,s->tmp,bufMV,bufL,bufU,zeros,s->maxIterations,1e-10);
 CHECK_HIP(hipStreamSynchronize(stream));printf("Total execution time of BiCGStab: %f (s)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());printf("Final error norm = %e\n",s->residual);
 for(Vec* v:vecs)CHECK_HIPSPARSE(hipsparseDestroyDnVec(v->vec));
 CHECK_HIPSPARSE(hipsparseDestroySpMat(s->a));CHECK_HIPSPARSE(hipsparseDestroySpMat(s->lower));CHECK_HIPSPARSE(hipsparseDestroySpMat(s->upper));CHECK_HIPSPARSE(hipsparseDestroy(s->sparse));CHECK_HIPBLAS(hipblasDestroy(s->blas));
 }
 return 0;
}
static void check(int result){if(result!=IRIS_SUCCESS){fprintf(stderr,"IRIS error %d\n",result);exit(2);}}
int main(int argc,char** argv){
 setvbuf(stdout,nullptr,_IOLBF,0);
 if(argc!=4){printf("Usage: %s <matrix.mtx> <maximum number of iterations> <verbose output>\n",argv[0]);return 1;}
 State s; s.maxIterations=atoi(argv[2]);s.verbose=atoi(argv[3]);if(s.maxIterations<=0)return 1;
 int lines,cols,symmetric,capacity;mtx_header(argv[1],&lines,&s.m,&cols,&capacity,&symmetric);
 if(s.m!=cols||!symmetric||s.m<=0)return 1;
 std::vector<int>rows(s.m+1),columns(capacity);std::vector<double>a(capacity);
 s.nnz=mtx_parsing(argv[1],lines,s.m,capacity,rows.data(),columns.data(),a.data(),0);a.resize(s.nnz);columns.resize(s.nnz);auto factors=a;
 std::vector<double>vectors(size_t(12)*s.m,0.);std::fill(vectors.begin()+s.m,vectors.begin()+2*s.m,1.);
 printf("matrix name: %s\nnum. rows: %d\nnum. cols: %d\nnnz: %d\n",argv[1],s.m,cols,s.nnz);
 if(!getenv("IRIS_KERNEL_HOST2HIP")){char path[4096];if(!realpath("libhecbench_vendor.so",path))return 2;setenv("IRIS_KERNEL_HOST2HIP",path,1);}
 iris::Platform platform;check(platform.init(nullptr,nullptr,true));int backend=-1;char name[256]={};iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_name,name,nullptr);if(backend!=iris_hip)return 2;printf("IRIS HIP backend=%d device=%s genuine hipSPARSE/hipBLAS\n",backend,name);
 {
 iris::DMem mr(rows.data(),rows.size()*4),mc(columns.data(),columns.size()*4),ma(a.data(),a.size()*8),mm(factors.data(),factors.size()*8),mv(vectors.data(),vectors.size()*8);
 State* state=&s;hecbench_vendor_callback callback=vendor;int phase=0;size_t one=1;
 void* p[]={&callback,&state,&phase,&mr,&mc,&ma,&mm,&mv};int modes[]={sizeof(callback),sizeof(state),sizeof(phase),iris_r,iris_r,iris_r,iris_rw,iris_rw};
 iris::Task setup;check(setup.kernel("hecbench_vendor",1,nullptr,&one,&one,8,p,modes));check(setup.submit(0,nullptr,true));
 std::vector<unsigned char>wm(std::max(size_t(1),s.bytesMV)),wi(std::max(1,s.bytesLU)),wl(std::max(size_t(1),s.bytesL)),wu(std::max(size_t(1),s.bytesU));
 iris::DMem mvm(wm.data(),wm.size()),mlu(wi.data(),wi.size()),ml(wl.data(),wl.size()),mu(wu.data(),wu.size());phase=1;
 void* q[]={&callback,&state,&phase,&mr,&mc,&ma,&mm,&mv,&mvm,&mlu,&ml,&mu};int qm[]={sizeof(callback),sizeof(state),sizeof(phase),iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw};
 iris::Task run;check(run.kernel("hecbench_vendor",1,nullptr,&one,&one,12,q,qm));check(run.flush_out(mv));check(run.flush_out(mm));check(run.submit(0,nullptr,true));
 }
 long double r2=0,b2=0;for(int i=0;i<s.m;++i){long double ax=0;for(int p=rows[i];p<rows[i+1];++p)ax+=static_cast<long double>(a[p])*vectors[s.m+columns[p]];long double r=vectors[i]-ax;r2+=r*r;b2+=static_cast<long double>(vectors[i])*vectors[i];}
 double residual=std::sqrt(r2),relative=std::sqrt(r2/b2);printf("Independent CPU residual = %.17g relative = %.17g\n",residual,relative);
 bool finite=std::isfinite(s.residual)&&std::isfinite(residual);for(int i=0;i<2*s.m;++i)finite=finite&&std::isfinite(vectors[i]);
 if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")){char path[4096];snprintf(path,sizeof(path),"%s.bin",prefix);FILE* f=fopen(path,"wb");if(!f)return 3;fwrite(vectors.data(),8,2*s.m,f);fwrite(factors.data(),8,s.nnz,f);fclose(f);}
 check(platform.finalize());return finite?0:1;
}
