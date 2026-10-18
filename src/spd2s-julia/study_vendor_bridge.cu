#define main original_spd2s_cli_main
#include "../spd2s-cuda/main.cu"
#undef main
#include <stdexcept>
struct StudySparse{cusparseHandle_t handle;cusparseDnMatDescr_t dense;cusparseSpMatDescr_t sparse;void*buffer;};
static void sparsecheck(cusparseStatus_t s){if(s!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error(cusparseGetErrorString(s));}
static void cudacheck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
extern "C" int study_sparse_version(){int version=0;cusparseHandle_t h;cusparseCreate(&h);cusparseGetVersion(h,&version);cusparseDestroy(h);return version;}
extern "C" void* study_prepare(float*dense,int64_t*rows,int64_t*cols,float*values,int64_t m,int64_t n,int64_t expected,int*status){
 try{auto*c=new StudySparse{};sparsecheck(cusparseCreate(&c->handle));sparsecheck(cusparseCreateDnMat(&c->dense,m,n,n,dense,CUDA_R_32F,CUSPARSE_ORDER_ROW));sparsecheck(cusparseCreateCsr(&c->sparse,m,n,0,rows,nullptr,nullptr,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_64I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));size_t bytes=0;sparsecheck(cusparseDenseToSparse_bufferSize(c->handle,c->dense,c->sparse,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,&bytes));cudacheck(cudaMalloc(&c->buffer,bytes));sparsecheck(cusparseDenseToSparse_analysis(c->handle,c->dense,c->sparse,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,c->buffer));int64_t mm,nn,nnz;sparsecheck(cusparseSpMatGetSize(c->sparse,&mm,&nn,&nnz));if(mm!=m||nn!=n||nnz!=expected)throw std::runtime_error("analysis shape/nnz mismatch");sparsecheck(cusparseCsrSetPointers(c->sparse,rows,cols,values));cudacheck(cudaDeviceSynchronize());*status=0;return c;}catch(const std::exception&e){fprintf(stderr,"SparsePrepare: %s\n",e.what());*status=1;return nullptr;}
}
extern "C" int study_convert(void*ctx){auto*c=(StudySparse*)ctx;return int(cusparseDenseToSparse_convert(c->handle,c->dense,c->sparse,CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,c->buffer));}
extern "C" int study_destroy(void*ctx){auto*c=(StudySparse*)ctx;cusparseDestroyDnMat(c->dense);cusparseDestroySpMat(c->sparse);cusparseDestroy(c->handle);cudaFree(c->buffer);delete c;return 0;}
