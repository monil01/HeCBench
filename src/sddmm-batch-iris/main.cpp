#include <iris/iris.hpp>
#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>
#include "../../tools/iris_vendor_bridge.hpp"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <unistd.h>
#include "utils.h"
#define GPU_CHECK(call) do{int status=(call);if(status){fprintf(stderr,"GPU API failure line %d code %d\n",__LINE__,status);return status;}}while(0)
struct State{
 int b,m,k,n,nnz,repeat;
 size_t bytes=0;
 hipsparseHandle_t handle=nullptr;
 std::vector<hipsparseDnMatDescr_t> a,bmat;
 std::vector<hipsparseSpMatDescr_t> c;
};
static int vendor(void* rawstream,unsigned,void* const* args){
 auto* s=hecbench_vendor_arg<State*>(args,1);int phase=hecbench_vendor_arg<int>(args,2);
 auto a=hecbench_vendor_arg<float*>(args,3), b=hecbench_vendor_arg<float*>(args,4),v=hecbench_vendor_arg<float*>(args,7);
 auto offsets=hecbench_vendor_arg<int*>(args,5),columns=hecbench_vendor_arg<int*>(args,6);
 auto stream=reinterpret_cast<hipStream_t>(rawstream);float alpha=1,beta=0;
 if(phase==0){
 GPU_CHECK(hipsparseCreate(&s->handle));GPU_CHECK(hipsparseSetStream(s->handle,stream));
 s->a.resize(s->b);s->bmat.resize(s->b);s->c.resize(s->b);
 for(int i=0;i<s->b;++i){
 GPU_CHECK(hipsparseCreateDnMat(&s->a[i],s->m,s->k,s->k,a+size_t(i)*s->m*s->k,HIP_R_32F,HIPSPARSE_ORDER_ROW));
 GPU_CHECK(hipsparseCreateDnMat(&s->bmat[i],s->k,s->n,s->n,b+size_t(i)*s->k*s->n,HIP_R_32F,HIPSPARSE_ORDER_ROW));
 GPU_CHECK(hipsparseCreateCsr(&s->c[i],s->m,s->n,s->nnz,offsets+size_t(i)*(s->m+1),columns+size_t(i)*s->nnz,v+size_t(i)*s->nnz,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_32I,HIPSPARSE_INDEX_BASE_ZERO,HIP_R_32F));
 }
 GPU_CHECK(hipsparseSDDMM_bufferSize(s->handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s->a[0],s->bmat[0],&beta,s->c[0],HIP_R_32F,HIPSPARSE_SDDMM_ALG_DEFAULT,&s->bytes));
 }else{
 void* scratch=hecbench_vendor_arg<void*>(args,8);
 GPU_CHECK(hipsparseSetStream(s->handle,stream));
 GPU_CHECK(hipsparseSDDMM_preprocess(s->handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s->a[0],s->bmat[0],&beta,s->c[0],HIP_R_32F,HIPSPARSE_SDDMM_ALG_DEFAULT,scratch));
 GPU_CHECK(hipStreamSynchronize(stream));auto start=std::chrono::steady_clock::now();
 for(int r=0;r<s->repeat;++r)for(int i=0;i<s->b;++i)
 GPU_CHECK(hipsparseSDDMM(s->handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s->a[i],s->bmat[i],&beta,s->c[i],HIP_R_32F,HIPSPARSE_SDDMM_ALG_DEFAULT,scratch));
 GPU_CHECK(hipStreamSynchronize(stream));printf("Average execution time of batched SDDMM (%d batches): %f (us)\n",s->b,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/s->repeat);
 for(int i=0;i<s->b;++i){GPU_CHECK(hipsparseDestroyDnMat(s->a[i]));GPU_CHECK(hipsparseDestroyDnMat(s->bmat[i]));GPU_CHECK(hipsparseDestroySpMat(s->c[i]));}
 GPU_CHECK(hipsparseDestroy(s->handle));
 }
 return 0;
}
static void check(int result){if(result!=IRIS_SUCCESS){fprintf(stderr,"IRIS error %d\n",result);exit(2);}}
int main(int argc,char** argv){
 if(argc!=8){printf("Usage %s <number of batches> <M> <K> <N> <number of non-zero elements> <repeat> <verify>\n",argv[0]);return 1;}
 State s; s.b=atoi(argv[1]);s.m=atoi(argv[2]);s.k=atoi(argv[3]);s.n=atoi(argv[4]);s.nnz=atoi(argv[5]);s.repeat=atoi(argv[6]);int verify=atoi(argv[7]);
 if(s.b<=0||s.m<=0||s.k<=0||s.n<=0||s.nnz<=0||s.repeat<=0||size_t(s.nnz)>size_t(s.m)*s.n)return 1;
 size_t as=size_t(s.m)*s.k,bs=size_t(s.k)*s.n,cs=size_t(s.m)*s.n;
 std::vector<float>a(size_t(s.b)*as),b(size_t(s.b)*bs),c(size_t(s.b)*cs),values(size_t(s.b)*s.nnz),reference(values.size());
 std::vector<int>offsets(size_t(s.b)*(s.m+1)),columns(values.size());
 for(int i=0;i<s.b;++i){init_matrix(a.data()+i*as,s.m,s.k,as);init_matrix(b.data()+i*bs,s.k,s.n,bs);init_matrix(c.data()+i*cs,s.m,s.n,s.nnz);init_csr(offsets.data()+size_t(i)*(s.m+1),values.data()+size_t(i)*s.nnz,columns.data()+size_t(i)*s.nnz,c.data()+i*cs,s.m,s.n,s.nnz);}
 if(verify)for(int i=0;i<s.b;++i)sddmm(a.data()+i*as,b.data()+i*bs,c.data()+i*cs,reference.data()+size_t(i)*s.nnz,offsets.data()+size_t(i)*(s.m+1),columns.data()+size_t(i)*s.nnz,s.k,s.m,s.n);
 if(!getenv("IRIS_KERNEL_HOST2HIP")){char path[4096];if(!realpath("libhecbench_vendor.so",path))return 2;setenv("IRIS_KERNEL_HOST2HIP",path,1);}
 iris::Platform platform;check(platform.init(nullptr,nullptr,true));int backend=-1;char name[256]={};iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_name,name,nullptr);if(backend!=iris_hip)return 2;printf("IRIS HIP backend=%d device=%s (genuine hipSPARSE)\n",backend,name);
 {
 iris::DMem ma(a.data(),a.size()*4),mb(b.data(),b.size()*4),mo(offsets.data(),offsets.size()*4),mc(columns.data(),columns.size()*4),mv(values.data(),values.size()*4);
 hecbench_vendor_callback callback=vendor;State* state=&s;int phase=0;size_t one=1;
 void* p[]={&callback,&state,&phase,&ma,&mb,&mo,&mc,&mv};int modes[]={sizeof(callback),sizeof(state),sizeof(phase),iris_r,iris_r,iris_r,iris_r,iris_rw};
 iris::Task setup;check(setup.kernel("hecbench_vendor",1,nullptr,&one,&one,8,p,modes));check(setup.submit(0,nullptr,true));
 std::vector<unsigned char>scratch(std::max(size_t(1),s.bytes));iris::DMem ms(scratch.data(),scratch.size());phase=1;
 void* q[]={&callback,&state,&phase,&ma,&mb,&mo,&mc,&mv,&ms};int qm[]={sizeof(callback),sizeof(state),sizeof(phase),iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw};
 iris::Task run;check(run.kernel("hecbench_vendor",1,nullptr,&one,&one,9,q,qm));check(run.flush_out(mv));check(run.submit(0,nullptr,true));
 }
 bool correct=true;if(verify){for(size_t i=0;i<values.size();++i)if(!std::isfinite(values[i])||fabsf(values[i]-reference[i])>1e-2f){correct=false;printf("@%zu %f != %f\n",i,values[i],reference[i]);break;}printf("sddmm_csr_batched_example test %s\n",correct?"PASSED":"FAILED: wrong result");}
 if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")){char path[4096];snprintf(path,sizeof(path),"%s.bin",prefix);FILE* f=fopen(path,"wb");if(!f)return 3;fwrite(values.data(),4,values.size(),f);fclose(f);}
 check(platform.finalize());return correct?0:1;
}
