#include <vector>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#define main original_bicgstab_main
#include "../bicgstab-cuda/main.cu"
#undef main
static int study_iterations;
#include "native_solver.inc"
struct Context {int m,nnz; int *rows,*cols; double *values,*factors; Vec v[11]; double *alloc[11]; cublasHandle_t blas; cusparseHandle_t sparse; cusparseSpMatDescr_t A,L,U; void *mv,*lb,*ub; cusparseSpSVDescr_t ld,ud;};
static Context c;
static constexpr double guard=91.25;
extern "C" int study_init(const char* file_path) {
    int base = 0;
    int num_rows, num_cols, nnz, num_lines, is_symmetric;
    mtx_header(file_path, &num_lines, &num_rows, &num_cols, &nnz, &is_symmetric);
    printf("\nmatrix name: %s\n"
           "num. rows:   %d\n"
           "num. cols:   %d\n"
           "nnz:         %d\n"
           "structure:   %s\n\n",
           file_path, num_rows, num_cols, nnz,
           (is_symmetric) ? "symmetric" : "unsymmetric");
    if (num_rows != num_cols) {
        printf("the input matrix must be square\n");
        return EXIT_FAILURE;
    }
    if (!is_symmetric) {
        printf("the input matrix must be symmetric\n");
        return EXIT_FAILURE;
    }
    int     m           = num_rows;
    int     num_offsets = m + 1;
    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
    printf("Matrix parsing...\n");
    nnz = mtx_parsing(file_path, num_lines, num_rows, nnz, h_A_rows,
                      h_A_columns, h_A_values, base);
    printf("Testing BiCGStab\n");
    for (int i = 0; i < num_rows; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // ### Device memory management ###
    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_M_values;
    Vec     d_B, d_X, d_R, d_R0, d_P, d_P_aux, d_S, d_S_aux, d_V, d_T, d_tmp;

    // allocate device memory for CSR matrices
    CHECK_CUDA( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_M_values,  nnz * sizeof(double)) )

    CHECK_CUDA( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_R0.ptr,    m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_P_aux.ptr, m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_S.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_S_aux.ptr, m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_V.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) )

    // copy the CSR matrices and vectors into device memory
    CHECK_CUDA( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_M_values, h_A_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_X.ptr, h_X, m * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // ### cuSPARSE Handle and descriptors initialization ###
    // create the test matrix on the host
    cublasHandle_t   cublasHandle   = NULL;
    cusparseHandle_t cusparseHandle = NULL;
    CHECK_CUBLAS( cublasCreate(&cublasHandle) )
    CHECK_CUSPARSE( cusparseCreate(&cusparseHandle) )
    // Create dense vectors
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_R0.vec,    m, d_R0.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_P.vec,     m, d_P.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_P_aux.vec, m, d_P_aux.ptr,
                                        CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_S.vec,     m, d_S.ptr, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_S_aux.vec, m, d_S_aux.ptr,
                                        CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_V.vec,   m, d_V.ptr,   CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F) )

    cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
    // IMPORTANT: Upper/Lower triangular decompositions of A
    //            (matM_lower, matM_upper) must use two distinct descriptors
    cusparseSpMatDescr_t matA, matM_lower, matM_upper;
    cusparseMatDescr_t   matLU;
    int*                 d_M_rows      = d_A_rows;
    int*                 d_M_columns   = d_A_columns;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // A
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                      d_A_columns, d_A_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    // M_lower
    CHECK_CUSPARSE( cusparseCreateCsr(&matM_lower, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)) )
    // M_upper
    CHECK_CUSPARSE( cusparseCreateCsr(&matM_upper, m, m, nnz, d_M_rows,
                                      d_M_columns, d_M_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      baseIdx, CUDA_R_64F) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)) )
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)) )
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    const double alpha = 0.75;
    size_t bufferSizeMV;
    void*  d_bufferMV;
    double beta = 0.0;
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) )
    CHECK_CUDA( cudaMalloc(&d_bufferMV, bufferSizeMV) )

    CHECK_CUSPARSE( cusparseSpMV(
                        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
    // X0 = 0
    CHECK_CUDA( cudaMemset(d_X.ptr, 0x0, m * sizeof(double)) )
    //--------------------------------------------------------------------------
    // Perform Incomplete-LU factorization of A (csrilu0) -> M_lower, M_upper
    csrilu02Info_t infoM        = NULL;
    int            bufferSizeLU = 0;
    void*          d_bufferLU;
    CHECK_CUSPARSE( cusparseCreateMatDescr(&matLU) )
    CHECK_CUSPARSE( cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL) )
    CHECK_CUSPARSE( cusparseSetMatIndexBase(matLU, baseIdx) )
    CHECK_CUSPARSE( cusparseCreateCsrilu02Info(&infoM) )

    CHECK_CUSPARSE( cusparseDcsrilu02_bufferSize(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM, &bufferSizeLU) )
    CHECK_CUDA( cudaMalloc(&d_bufferLU, bufferSizeLU) )
    CHECK_CUSPARSE( cusparseDcsrilu02_analysis(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) )
    int structural_zero;
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM,
                                                &structural_zero) )
    // M = L * U
    CHECK_CUSPARSE( cusparseDcsrilu02(
                        cusparseHandle, m, nnz, matLU, d_M_values,
                        d_A_rows, d_A_columns, infoM,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU) )
    // Find numerical zero
    int numerical_zero;
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot(cusparseHandle, infoM,
                                                &numerical_zero) )

    CHECK_CUSPARSE( cusparseDestroyCsrilu02Info(infoM) )
    CHECK_CUSPARSE( cusparseDestroyMatDescr(matLU) )
    CHECK_CUDA( cudaFree(d_bufferLU) )
    //--------------------------------------------------------------------------
    // ### Run BiCGStab computation ###

 c.m=m;c.nnz=nnz;c.rows=d_A_rows;c.cols=d_A_columns;c.values=d_A_values;c.factors=d_M_values;c.blas=cublasHandle;c.sparse=cusparseHandle;c.A=matA;c.L=matM_lower;c.U=matM_upper;c.mv=d_bufferMV;
 Vec vv[]={d_B,d_X,d_R0,d_R,d_P,d_P_aux,d_S,d_S_aux,d_V,d_T,d_tmp};
 for(int k=0;k<11;k++) {c.v[k]=vv[k];std::vector<double> old(m);if(k==0){CHECK_CUDA(cudaMemcpy(old.data(),vv[k].ptr,m*8,cudaMemcpyDeviceToHost));}CHECK_CUDA(cudaFree(vv[k].ptr));CHECK_CUDA(cudaMalloc(&c.alloc[k],(m+32)*8));std::vector<double> a(m+32,guard);std::fill(a.begin()+16,a.end()-16,0.);if(k==0)std::copy(old.begin(),old.end(),a.begin()+16);CHECK_CUDA(cudaMemcpy(c.alloc[k],a.data(),a.size()*8,cudaMemcpyHostToDevice));c.v[k].ptr=c.alloc[k]+16;CHECK_CUSPARSE(cusparseDnVecSetValues(c.v[k].vec,c.v[k].ptr));}
 free(h_A_rows);free(h_A_columns);free(h_A_values);free(h_X);return 0;
}
extern "C" int study_n(){return c.m;}
extern "C" int study_nnz(){return c.nnz;}
extern "C" int study_sync(){return (int)cudaDeviceSynchronize();}
extern "C" int study_reset(){for(int k=1;k<11;k++){CHECK_CUDA(cudaMemset(c.v[k].ptr,0,c.m*8));}study_iterations=0;return 0;}
extern "C" int study_capture(double* out){
 std::vector<int> ii(c.m+1),jj(c.nnz);CHECK_CUDA(cudaMemcpy(ii.data(),c.rows,ii.size()*4,cudaMemcpyDeviceToHost));CHECK_CUDA(cudaMemcpy(jj.data(),c.cols,jj.size()*4,cudaMemcpyDeviceToHost));for(auto x:ii)*out++=x;for(auto x:jj)*out++=x;
 CHECK_CUDA(cudaMemcpy(out,c.values,c.nnz*8,cudaMemcpyDeviceToHost));out+=c.nnz;CHECK_CUDA(cudaMemcpy(out,c.factors,c.nnz*8,cudaMemcpyDeviceToHost));out+=c.nnz;
 for(int k=0;k<11;k++){CHECK_CUDA(cudaMemcpy(out,c.alloc[k],(c.m+32)*8,cudaMemcpyDeviceToHost));out+=c.m+32;}return 0;
}
extern "C" int study_capture_count(){return c.m+1+3*c.nnz+11*(c.m+32);}
extern "C" double study_native(){return study_native_solver(0,c.blas,c.sparse,c.m,c.A,c.L,c.U,c.v[0],c.v[1],c.v[2],c.v[3],c.v[4],c.v[5],c.v[6],c.v[7],c.v[8],c.v[9],c.v[10],c.mv,100,1e-10);}
extern "C" int study_last_iterations(){return study_iterations;}
extern "C" int study_prepare(){
 const double coefficient=1.;size_t l,u;
 CHECK_CUSPARSE(cusparseSpSV_createDescr(&c.ld));
 CHECK_CUSPARSE(cusparseSpSV_bufferSize(c.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&coefficient,c.L,c.v[4].vec,c.v[10].vec,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,c.ld,&l));CHECK_CUDA(cudaMalloc(&c.lb,l));
 CHECK_CUSPARSE(cusparseSpSV_analysis(c.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&coefficient,c.L,c.v[4].vec,c.v[10].vec,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,c.ld,c.lb));
 CHECK_CUSPARSE(cusparseSpSV_createDescr(&c.ud));
 CHECK_CUSPARSE(cusparseSpSV_bufferSize(c.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&coefficient,c.U,c.v[10].vec,c.v[5].vec,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,c.ud,&u));CHECK_CUDA(cudaMalloc(&c.ub,u));
 CHECK_CUSPARSE(cusparseSpSV_analysis(c.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&coefficient,c.U,c.v[10].vec,c.v[5].vec,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,c.ud,c.ub));return 0;
}
extern "C" int study_finish(){CHECK_CUSPARSE(cusparseSpSV_destroyDescr(c.ld));CHECK_CUSPARSE(cusparseSpSV_destroyDescr(c.ud));CHECK_CUDA(cudaFree(c.lb));CHECK_CUDA(cudaFree(c.ub));return 0;}
extern "C" int study_copy(int src,int dst){CHECK_CUDA(cudaMemcpy(c.v[dst].ptr,c.v[src].ptr,c.m*8,cudaMemcpyDeviceToDevice));return 0;}
extern "C" int study_zero(int v){CHECK_CUDA(cudaMemset(c.v[v].ptr,0,c.m*8));return 0;}
extern "C" double study_dot(int x,int y){double a;auto e=cublasDdot(c.blas,c.m,c.v[x].ptr,1,c.v[y].ptr,1,&a);if(e) return NAN;return a;}
extern "C" double study_norm(int x){double a;auto e=cublasDnrm2(c.blas,c.m,c.v[x].ptr,1,&a);if(e)return NAN;return a;}
extern "C" int study_axpy(int x,int y,double a){CHECK_CUBLAS(cublasDaxpy(c.blas,c.m,&a,c.v[x].ptr,1,c.v[y].ptr,1));return 0;}
extern "C" int study_scale(int x,double a){CHECK_CUBLAS(cublasDscal(c.blas,c.m,&a,c.v[x].ptr,1));return 0;}
extern "C" int study_mv(int x,int y,double a,double b){CHECK_CUSPARSE(cusparseSpMV(c.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&a,c.A,c.v[x].vec,&b,c.v[y].vec,CUDA_R_64F,CUSPARSE_SPMV_ALG_DEFAULT,c.mv));return 0;}
extern "C" int study_solve(int upper,int x,int y){double a=1.;CHECK_CUSPARSE(cusparseSpSV_solve(c.sparse,CUSPARSE_OPERATION_NON_TRANSPOSE,&a,upper?c.U:c.L,c.v[x].vec,c.v[y].vec,CUDA_R_64F,CUSPARSE_SPSV_ALG_DEFAULT,upper?c.ud:c.ld));return 0;}
extern "C" const char* study_library(int k){Dl_info d;dladdr(k?(void*)cusparseCreate:(void*)cublasCreate_v2,&d);return d.dli_fname;}
extern "C" int study_version(int k){int v; if(k)cusparseGetVersion(c.sparse,&v);else cublasGetVersion(c.blas,&v);return v;}

extern "C" int study_native_timed(double* seconds,double* residual,int* iterations){CHECK_CUDA(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();*residual=study_native();CHECK_CUDA(cudaDeviceSynchronize());*seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();*iterations=study_iterations;return 0;}
extern "C" int study_destroy(){for(int k=0;k<11;k++){CHECK_CUSPARSE(cusparseDestroyDnVec(c.v[k].vec));CHECK_CUDA(cudaFree(c.alloc[k]));}for(auto a:{c.A,c.L,c.U}){CHECK_CUSPARSE(cusparseDestroySpMat(a));}CHECK_CUDA(cudaFree(c.rows));CHECK_CUDA(cudaFree(c.cols));CHECK_CUDA(cudaFree(c.values));CHECK_CUDA(cudaFree(c.factors));CHECK_CUDA(cudaFree(c.mv));CHECK_CUSPARSE(cusparseDestroy(c.sparse));CHECK_CUBLAS(cublasDestroy(c.blas));return 0;}
// Independent coordinate stencil/known solution and row-elimination ILU0 checks,
// outside ROI. No cuBLAS/cuSPARSE oracle is used.
extern "C" int study_check(const double* before,const double* after,double norm,int iterations,double* metrics){
 const int n=c.m,e=c.nnz;int off=n+1+3*e;bool ok=n==1024&&iterations>1&&iterations<=100;double ma=0,mr=0,res2=0;auto v=[&](const double* a,int k,int i){return a[off+k*(n+32)+16+i];};
 for(int i=0;i<study_capture_count();i++)ok=ok&&std::isfinite(before[i])&&std::isfinite(after[i]);
 for(int i=0;i<off;i++)ok=ok&&before[i]==after[i];
 for(int k=0;k<11;k++){for(int j=0;j<16;j++)ok=ok&&before[off+k*(n+32)+j]==guard&&after[off+k*(n+32)+j]==guard&&before[off+k*(n+32)+n+16+j]==guard&&after[off+k*(n+32)+n+16+j]==guard;}
 int pos=0;std::vector<int> rows(n+1),cols;std::vector<double> vals,factors;
 for(int i=0;i<n;i++){rows[i]=pos;std::vector<int> js;if(i>=32)js.push_back(i-32);if(i%32)js.push_back(i-1);js.push_back(i);if(i%32!=31)js.push_back(i+1);if(i<992)js.push_back(i+32);for(int j:js){cols.push_back(j);vals.push_back(j==i?4.25:-1.);pos++;}}
 rows[n]=pos;ok=ok&&pos==e;for(int i=0;i<=n;i++)ok=ok&&before[i]==rows[i];for(int k=0;k<e;k++)ok=ok&&before[n+1+k]==cols[k]&&before[n+1+e+k]==vals[k];
 factors=vals;for(int i=0;i<n;i++){for(int k=rows[i];k<rows[i+1]&&cols[k]<i;k++){int j=cols[k],diag=rows[j];while(cols[diag]!=j)diag++;factors[k]/=factors[diag];for(int q=diag+1;q<rows[j+1];q++){auto it=std::lower_bound(cols.begin()+k+1,cols.begin()+rows[i+1],cols[q]);if(it!=cols.begin()+rows[i+1]&&*it==cols[q])factors[it-cols.begin()]-=factors[k]*factors[q];}}}
 for(int k=0;k<e;k++)ok=ok&&std::abs(before[n+1+2*e+k]-factors[k])<1e-12;
 for(int i=0;i<n;i++){
  int degree=(i>=32)+(i<992)+(i%32!=0)+(i%32!=31);double rhs=.75*(4.25-degree);ok=ok&&v(before,0,i)==rhs&&v(after,0,i)==rhs;
  for(int k=1;k<11;k++)ok=ok&&v(before,k,i)==0.;
  double x=v(after,1,i),err=std::abs(x-.75);ma=std::max(ma,err);mr=std::max(mr,err/.75);
  long double ax=4.25L*x;if(i>=32)ax-=v(after,1,i-32);if(i<992)ax-=v(after,1,i+32);if(i%32)ax-=v(after,1,i-1);if(i%32!=31)ax-=v(after,1,i+1);double residual=(double)((long double)rhs-ax);res2+=residual*residual;ok=ok&&std::abs(residual-v(after,3,i))<1e-12;
 }
 double rn=std::sqrt(res2);ok=ok&&ma<=2e-8&&rn<=2e-9&&std::abs(rn-norm)<=1e-12;metrics[0]=ma;metrics[1]=mr;metrics[2]=rn;return ok?1:0;
}
