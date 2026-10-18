#include "../sptrsv-sycl/sptrsv.h"
#include "../sptrsv-sycl/mmio.h"
#include "../sptrsv-sycl/tranpose.h"
#include "../sptrsv-sycl/read_mtx.h"
#include <fstream>
#include <stdexcept>
template<class T>void save(const char*n,T*p,int z){std::ofstream f(n,std::ios::binary);f.write((char*)p,z*sizeof(T));if(!f)throw std::runtime_error("write");}
int main(int argc,char**argv){if(argc!=2)return 2;int m,n,nnz,*rp,*ci;double*v; if(read_mtx(argv[1],&m,&n,&nnz,&rp,&ci,&v))return 3;int nz,*lr,*lc;double*lv;change2tran(m,nnz,rp,ci,v,&nz,&lr,&lc,&lv);double*x,*b;get_x_b(m,m,lr,lc,lv,&x,&b);int *map=new int[m+1],len;double wo,eo;matrix_warp4(m,m,nz,lr,lc,lv,10,&len,map,&wo,&eo);save("lp1_rp.i32",lr,m+1);save("lp1_ci.i32",lc,nz);save("lp1_val.f64",lv,nz);save("lp1_b.f64",b,m);save("lp1_x.f64",x,m);save("lp1_map.i32",map,len);printf("SOURCE_EXTRACT m=%d nnz=%d len=%d warpoccupancy=%.9g elementoccupancy=%.9g\n",m,nz,len,wo,eo);}
