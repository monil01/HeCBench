from pathlib import Path
import hashlib,json
p=Path('src/bicgstab-julia');s=Path('src/bicgstab-cuda/main.cu').read_text()
f=s[s.index('double gpu_BiCGStab('):s.index('//==============================================================================',s.index('double gpu_BiCGStab('))]
f=f.replace('double              coeff_tmp;', 'double              coeff_tmp = 1.0; // defined named analysis coefficient').replace('double gpu_BiCGStab(', 'double study_native_solver(').replace('for (int i = 1; i <= maxIterations; i++) {','for (int i = 1; i <= maxIterations; i++) {\n        study_iterations = i; // named measurement instrumentation only')
init=s[s.index('    int base = 0;',s.index('int main(')):s.index('    printf("BiCGStab loop:',s.index('int main('))]
(p/'native_solver.inc').write_text('// Source-exact arithmetic; one iteration-counter assignment added.\n'+f)
(p/'native_extraction.json').write_text(json.dumps({'source':'src/bicgstab-cuda/main.cu','source_sha256':hashlib.sha256(s.encode()).hexdigest(),'function_original_sha256':hashlib.sha256(s[s.index('double gpu_BiCGStab('):s.index('//==============================================================================',s.index('double gpu_BiCGStab('))].encode()).hexdigest(),'extracted_sha256':hashlib.sha256((p/'native_solver.inc').read_bytes()).hexdigest(),'modifications':['function renamed study_native_solver','initialize otherwise uninitialized SpSV analysis coefficient to1 (actual solve coefficient already1); conditional repaired host reference','one host iteration-counter assignment inside existing loop; no solver recurrence/API changes'],'original_cli_run':'.porting-state/diagnostics/bicgstab/original_run.log','fixture':'deterministic32x32shift0.25SPD','original_external_TEM_dataset':'absent/unverified'},indent=2)+'\n')
header='''#include <vector>
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
'''
finish='''
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
'''
finish+='''
extern "C" int study_native_timed(double* seconds,double* residual,int* iterations){CHECK_CUDA(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();*residual=study_native();CHECK_CUDA(cudaDeviceSynchronize());*seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();*iterations=study_iterations;return 0;}
extern "C" int study_destroy(){for(int k=0;k<11;k++){CHECK_CUSPARSE(cusparseDestroyDnVec(c.v[k].vec));CHECK_CUDA(cudaFree(c.alloc[k]));}for(auto a:{c.A,c.L,c.U}){CHECK_CUSPARSE(cusparseDestroySpMat(a));}CHECK_CUDA(cudaFree(c.rows));CHECK_CUDA(cudaFree(c.cols));CHECK_CUDA(cudaFree(c.values));CHECK_CUDA(cudaFree(c.factors));CHECK_CUDA(cudaFree(c.mv));CHECK_CUSPARSE(cusparseDestroy(c.sparse));CHECK_CUBLAS(cublasDestroy(c.blas));return 0;}
'''
(p/'study_bridge.cu').write_text(header+init+finish+'// Independent coordinate stencil/known solution and row-elimination ILU0 checks,\n// outside ROI. No cuBLAS/cuSPARSE oracle is used.\nextern "C" int study_check(const double* before,const double* after,double norm,int iterations,double* metrics){\n const int n=c.m,e=c.nnz;int off=n+1+3*e;bool ok=n==1024&&iterations>1&&iterations<=100;double ma=0,mr=0,res2=0;auto v=[&](const double* a,int k,int i){return a[off+k*(n+32)+16+i];};\n for(int i=0;i<study_capture_count();i++)ok=ok&&std::isfinite(before[i])&&std::isfinite(after[i]);\n for(int i=0;i<off;i++)ok=ok&&before[i]==after[i];\n for(int k=0;k<11;k++){for(int j=0;j<16;j++)ok=ok&&before[off+k*(n+32)+j]==guard&&after[off+k*(n+32)+j]==guard&&before[off+k*(n+32)+n+16+j]==guard&&after[off+k*(n+32)+n+16+j]==guard;}\n int pos=0;std::vector<int> rows(n+1),cols;std::vector<double> vals,factors;\n for(int i=0;i<n;i++){rows[i]=pos;std::vector<int> js;if(i>=32)js.push_back(i-32);if(i%32)js.push_back(i-1);js.push_back(i);if(i%32!=31)js.push_back(i+1);if(i<992)js.push_back(i+32);for(int j:js){cols.push_back(j);vals.push_back(j==i?4.25:-1.);pos++;}}\n rows[n]=pos;ok=ok&&pos==e;for(int i=0;i<=n;i++)ok=ok&&before[i]==rows[i];for(int k=0;k<e;k++)ok=ok&&before[n+1+k]==cols[k]&&before[n+1+e+k]==vals[k];\n factors=vals;for(int i=0;i<n;i++){for(int k=rows[i];k<rows[i+1]&&cols[k]<i;k++){int j=cols[k],diag=rows[j];while(cols[diag]!=j)diag++;factors[k]/=factors[diag];for(int q=diag+1;q<rows[j+1];q++){auto it=std::lower_bound(cols.begin()+k+1,cols.begin()+rows[i+1],cols[q]);if(it!=cols.begin()+rows[i+1]&&*it==cols[q])factors[it-cols.begin()]-=factors[k]*factors[q];}}}\n for(int k=0;k<e;k++)ok=ok&&std::abs(before[n+1+2*e+k]-factors[k])<1e-12;\n for(int i=0;i<n;i++){\n  int degree=(i>=32)+(i<992)+(i%32!=0)+(i%32!=31);double rhs=.75*(4.25-degree);ok=ok&&v(before,0,i)==rhs&&v(after,0,i)==rhs;\n  for(int k=1;k<11;k++)ok=ok&&v(before,k,i)==0.;\n  double x=v(after,1,i),err=std::abs(x-.75);ma=std::max(ma,err);mr=std::max(mr,err/.75);\n  long double ax=4.25L*x;if(i>=32)ax-=v(after,1,i-32);if(i<992)ax-=v(after,1,i+32);if(i%32)ax-=v(after,1,i-1);if(i%32!=31)ax-=v(after,1,i+1);double residual=(double)((long double)rhs-ax);res2+=residual*residual;ok=ok&&std::abs(residual-v(after,3,i))<1e-12;\n }\n double rn=std::sqrt(res2);ok=ok&&ma<=2e-8&&rn<=2e-9&&std::abs(rn-norm)<=1e-12;metrics[0]=ma;metrics[1]=mr;metrics[2]=rn;return ok?1:0;\n}\n')
