#include <iris/iris.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "eigen.h"
#include "matrix.h"
#include "lanczos.h"
#include "cycle_timer.h"
#include "validation.h"

static void checked(int rc) { if(rc!=IRIS_SUCCESS) throw std::runtime_error("IRIS Lanczos task failed"); }
template<class T> struct DeviceLanczos {
  std::string suffix=std::is_same<T,float>::value ? "f32" : "f64";
  int rows, blocks;
  size_t gws[1],lws[1]={256};
  std::vector<T> a,b,c,parts;
  std::unique_ptr<iris::DMem> x,prev,y;
  iris::DMem rp,ci,values,scratch;
  DeviceLanczos(const csr_matrix<T>& m,const std::vector<T>& v)
    : rows(m.row_size()),blocks((rows+255)/256),gws{size_t(blocks)*256},a(v),b(rows),c(rows),parts(blocks),
      rp(const_cast<int*>(m.row_ptr_data()),sizeof(int)*size_t(rows+1)),
      ci(const_cast<int*>(m.col_ind_data()),sizeof(int)*size_t(m.nonzeros())),
      values(const_cast<T*>(m.values_data()),sizeof(T)*size_t(m.nonzeros())),scratch(parts.data(),sizeof(T)*parts.size()) {
    x.reset(new iris::DMem(a.data(),sizeof(T)*a.size()));
    prev.reset(new iris::DMem(b.data(),sizeof(T)*b.size()));
    y.reset(new iris::DMem(c.data(),sizeof(T)*c.size()));
  }
  T dot(iris::DMem& one,iris::DMem& two) {
    iris::Task task; void* args[]={&rows,&one,&two,&scratch}; int info[]={sizeof(rows),iris_r,iris_r,iris_w};
    checked(task.kernel(("dot_"+suffix).c_str(),1,nullptr,gws,lws,4,args,info));
    checked(task.flush_out(scratch)); checked(task.submit(iris_gpu,nullptr,1));
    T sum(0); for(int j=0;j<blocks;++j) sum+=parts[j]; return sum;
  }
  void saxpy(iris::DMem& out,iris::DMem& in,T scale) {
    iris::Task task; void* args[]={&rows,&out,&in,&scale}; int info[]={sizeof(rows),iris_rw,iris_r,sizeof(scale)};
    checked(task.kernel(("saxpy_"+suffix).c_str(),1,nullptr,gws,lws,4,args,info)); checked(task.submit(iris_gpu,nullptr,1));
  }
  void multiply(iris::DMem& out,T scale) {
    iris::Task task; void* args[]={&rows,&out,&scale}; int info[]={sizeof(rows),iris_rw,sizeof(scale)};
    checked(task.kernel(("mul_"+suffix).c_str(),1,nullptr,gws,lws,3,args,info)); checked(task.submit(iris_gpu,nullptr,1));
  }
  void spmv(int group,int multiply_blocks) {
    iris::Task task; int begin=0; size_t work[]={size_t(multiply_blocks)*256};
    void* args[]={&group,&rows,&begin,&rp,&ci,&values,x.get(),y.get()};
    int info[]={sizeof(group),sizeof(rows),sizeof(begin),iris_r,iris_r,iris_r,iris_r,iris_w};
    checked(task.kernel(("spmv_"+suffix).c_str(),1,nullptr,work,lws,8,args,info)); checked(task.submit(iris_gpu,nullptr,1));
  }
  std::vector<T>& host(iris::DMem* mem) {
    if(mem->host()==a.data()) return a;
    if(mem->host()==b.data()) return b;
    return c;
  }
};

template<class T> symm_tridiag_matrix<T> gpu_lanczos(const csr_matrix<T>& m,const std::vector<T>& v,int steps) {
  symm_tridiag_matrix<T> result(steps+1);
  int rows=m.row_size(),cols=m.col_size(),nonzeros=m.nonzeros();
  assert(rows==cols); assert(cols==int(v.size()));
  DeviceLanczos<T> dev(m,v);
  int row_nonzeros=nonzeros/rows;
  int group=row_nonzeros>16 ? 32 : 16;
  group=row_nonzeros>8 ? group : 8;
  group=row_nonzeros>4 ? group : 4;
  group=row_nonzeros>2 ? group : 2;
  int groups_per_block=256/group, multiply_blocks=(rows+groups_per_block-1)/groups_per_block;
  double start=cycle_timer::current_seconds();
  for(int i=0;i<steps;++i) {
    dev.spmv(group,multiply_blocks);
    T product=dev.dot(*dev.x,*dev.y); result.alpha(i)=product;
    dev.saxpy(*dev.y,*dev.x,-product);
    if(i>0) dev.saxpy(*dev.y,*dev.prev,-result.beta(i-1));
    dev.x.swap(dev.prev);
    result.beta(i)=T(sqrt(dev.dot(*dev.y,*dev.y)));
    dev.multiply(*dev.y,1/result.beta(i));
    dev.x.swap(dev.y);
  }
  double end=cycle_timer::current_seconds();
  std::cout<<"GPU Lanczos iterations: "<<steps<<std::endl;
  std::cout<<"GPU Lanczos time: "<<end-start<<" sec"<<std::endl;
  if(getenv("LANCZOS_DUMP_VECTORS")) {
    iris::Task flush;
    checked(flush.flush_out(*dev.x)); checked(flush.flush_out(*dev.prev)); checked(flush.flush_out(*dev.y)); checked(flush.submit(iris_gpu,nullptr,1));
  }
  LanczosObserver<T>::capture(steps,result,dev.host(dev.x.get()),dev.host(dev.prev.get()),dev.host(dev.y.get()));
  result.resize(steps); return result;
}
template<class T> std::vector<T> gpu_lanczos_eigen(const csr_matrix<T>& matrix,int k,int steps) {
  int cols=matrix.col_size(); assert(cols>0);
  std::vector<T> v(cols,0); v[0]=1;
  auto tridiag=gpu_lanczos(matrix,v,steps);
  return lanczos_no_spurious(tridiag,k);
}
template std::vector<float> gpu_lanczos_eigen(const csr_matrix<float>&,int,int);
template std::vector<double> gpu_lanczos_eigen(const csr_matrix<double>&,int,int);
