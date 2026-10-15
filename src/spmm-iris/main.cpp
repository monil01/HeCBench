#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <climits>
#include "utils.h"
static void check(int code){if(code!=IRIS_SUCCESS){fprintf(stderr,"IRIS error %d\n",code);exit(2);}}
static void launch(const char* name,int device,size_t global,size_t local,int count,void** args,int* modes){
 iris::Task t;check(t.kernel(name,1,nullptr,&global,&local,count,args,modes));check(t.submit(device,nullptr,true));
}
int main(int argc,char** argv){
 if(argc!=8){printf("Usage %s <M> <K> <N> <A_nnz> <B_nnz> <repeat> <verify>\n",argv[0]);return 1;}
 int m=atoi(argv[1]),k=atoi(argv[2]),n=atoi(argv[3]),an=atoi(argv[4]),bn=atoi(argv[5]),repeat=atoi(argv[6]),verify=atoi(argv[7]);
 if(m<=0||k<=0||n<=0||an<=0||bn<=0||repeat<=0||size_t(an)>size_t(m)*k||size_t(bn)>size_t(k)*n||size_t(m)*n>INT_MAX)return 1;
 std::vector<float>a(size_t(m)*k),b(size_t(k)*n),av(an),bv(bn),scratch(size_t(m)*n),cv(size_t(m)*n);
 std::vector<int>ao(m+1),ac(an),bo(k+1),bc(bn),counts(m),co(m+1),cc(size_t(m)*n);
 init_matrix(a.data(),m,k,an);init_csr(ao.data(),av.data(),ac.data(),a.data(),m,k,an);
 init_matrix(b.data(),k,n,bn);init_csr(bo.data(),bv.data(),bc.data(),b.data(),k,n,bn);
 iris::Platform platform;check(platform.init(nullptr,nullptr,true));
 int device=-1,backend=-1;for(int d=0;d<platform.ndevs();++d){check(iris_device_info(d,iris_backend,&backend,nullptr));if(backend==iris_hip){device=d;break;}}
 if(device<0)return 2;
 char name[256]={};check(iris_device_info(device,iris_name,name,nullptr));printf("IRIS HIP backend=%d device=%s\n",backend,name);
 {
 iris::DMem mao(ao.data(),ao.size()*4),mac(ac.data(),ac.size()*4),mav(av.data(),av.size()*4),mbo(bo.data(),bo.size()*4),mbc(bc.data(),bc.size()*4),mbv(bv.data(),bv.size()*4);
 iris::DMem ms(scratch.data(),scratch.size()*4),mcnt(counts.data(),counts.size()*4),mco(co.data(),co.size()*4),mcc(cc.data(),cc.size()*4),mcv(cv.data(),cv.size()*4);
 size_t local=64,global=(size_t(m)+local-1)/local*local;
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<repeat;++i){
 void* p[]={&mao,&mac,&mav,&mbo,&mbc,&mbv,&ms,&mcnt,&m,&n};
 int modes[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,iris_w,sizeof(m),sizeof(n)};
 launch("product",device,global,local,10,p,modes);
 }
 auto end=std::chrono::steady_clock::now();
 printf("Average execution time of SPMM compute: %f (us)\n",std::chrono::duration<double,std::micro>(end-start).count()/repeat);
 void* p[]={&mcnt,&mco,&m};int modes[]={iris_r,iris_w,sizeof(m)};launch("prefix",device,1,1,3,p,modes);
 iris::Task t;void* e[]={&ms,&mco,&mcc,&mcv,&m,&n};int em[]={iris_r,iris_r,iris_w,iris_w,sizeof(m),sizeof(n)};
 check(t.kernel("emit",1,nullptr,&global,&local,6,e,em));check(t.flush_out(mco));check(t.flush_out(mcc));check(t.flush_out(mcv));check(t.submit(device,nullptr,true));
 }
 int nnz=co[m];printf("Output CSR nnz: %d\n",nnz);bool correct=true;
 if(verify){
 float* rv=nullptr;int* ro=nullptr;int* rc=nullptr;int rn=spmm(a.data(),b.data(),rv,ro,rc,k,m,n);
 correct=rn==nnz;
 for(int i=0;i<=m;++i)correct=correct && ro[i]==co[i];
 if(correct)for(int i=0;i<nnz;++i)correct=correct && rc[i]==cc[i] && std::isfinite(cv[i]) && fabsf(rv[i]-cv[i])<=1e-2f;
 free(rv);free(ro);free(rc);printf("spgemm_example test %s\n",correct?"PASSED":"FAILED: wrong result");
 }
 if(const char* prefix=getenv("HECBENCH_DUMP_PREFIX")){
 char file[4096];snprintf(file,sizeof(file),"%s.bin",prefix);FILE* f=fopen(file,"wb");if(!f)return 3;
 fwrite(co.data(),4,m+1,f);fwrite(cc.data(),4,nnz,f);fwrite(cv.data(),4,nnz,f);fclose(f);
 }
 check(platform.finalize());return correct?0:1;
}
