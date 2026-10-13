#include <iris/iris.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
extern int gpu_device;
long ccsd_tengy_gpu(const double * __restrict__ f1n,    const double * __restrict__ f1t,
                    const double * __restrict__ f2n,    const double * __restrict__ f2t,
                    const double * __restrict__ f3n,    const double * __restrict__ f3t,
                    const double * __restrict__ f4n,    const double * __restrict__ f4t,
                    const double * __restrict__ dintc1, const double * __restrict__ dintx1, const double * __restrict__ t1v1,
                    const double * __restrict__ dintc2, const double * __restrict__ dintx2, const double * __restrict__ t1v2,
                    const double * __restrict__ eorb,   const double eaijk,
                    double * __restrict__ emp4i_, double * __restrict__ emp5i_,
                    double * __restrict__ emp4k_, double * __restrict__ emp5k_,
                    const int ncor, const int nocc, const int nvir)
{
  double emp5i = 0.0, emp4i = 0.0, emp5k = 0.0, emp4k = 0.0;

  iris::DMem d_f1n(const_cast<double*>(f1n),nvir*nvir*sizeof(double));
  iris::DMem d_f2n(const_cast<double*>(f2n),nvir*nvir*sizeof(double));
  iris::DMem d_f3n(const_cast<double*>(f3n),nvir*nvir*sizeof(double));
  iris::DMem d_f4n(const_cast<double*>(f4n),nvir*nvir*sizeof(double));
  iris::DMem d_f1t(const_cast<double*>(f1t),nvir*nvir*sizeof(double));
  iris::DMem d_f2t(const_cast<double*>(f2t),nvir*nvir*sizeof(double));
  iris::DMem d_f3t(const_cast<double*>(f3t),nvir*nvir*sizeof(double));
  iris::DMem d_f4t(const_cast<double*>(f4t),nvir*nvir*sizeof(double));
  iris::DMem d_dintc1(const_cast<double*>(dintc1),nvir*sizeof(double));
  iris::DMem d_dintc2(const_cast<double*>(dintc2),nvir*sizeof(double));
  iris::DMem d_dintx1(const_cast<double*>(dintx1),nvir*sizeof(double));
  iris::DMem d_dintx2(const_cast<double*>(dintx2),nvir*sizeof(double));
  iris::DMem d_t1v1(const_cast<double*>(t1v1),nvir*sizeof(double));
  iris::DMem d_t1v2(const_cast<double*>(t1v2),nvir*sizeof(double));
  iris::DMem d_eorb(const_cast<double*>(eorb),(ncor+nocc+nvir)*sizeof(double));
  iris::DMem d_emp5i(&emp5i,sizeof(double));
  iris::DMem d_emp4i(&emp4i,sizeof(double));
  iris::DMem d_emp5k(&emp5k,sizeof(double));
  iris::DMem d_emp4k(&emp4k,sizeof(double));


  auto t0 = std::chrono::steady_clock::now();

  iris::Task task;
  void* params[]={&d_f1n,&d_f1t,&d_f2n,&d_f2t,&d_f3n,&d_f3t,&d_f4n,&d_f4t,&d_dintc1,&d_dintx1,&d_t1v1,&d_dintc2,&d_dintx2,&d_t1v2,&d_eorb,const_cast<double*>(&eaijk),
    &d_emp4i,&d_emp5i,&d_emp4k,&d_emp5k,
    const_cast<int*>(&ncor),const_cast<int*>(&nocc),const_cast<int*>(&nvir)};
  int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,sizeof(eaijk),
    iris_rw,iris_rw,iris_rw,iris_rw,sizeof(ncor),sizeof(nocc),sizeof(nvir)};
  size_t local[]={16,16},global[]={size_t((nvir+15)/16)*16,size_t((nvir+15)/16)*16};
  if(task.kernel("ccsd_kernel",2,nullptr,global,local,23,params,info))throw std::runtime_error("CCSD setup failed");
  task.flush_out(d_emp4i);task.flush_out(d_emp5i);task.flush_out(d_emp4k);task.flush_out(d_emp5k);
  if(task.submit(gpu_device,nullptr,1))throw std::runtime_error("CCSD kernel failed");

  auto t1 = std::chrono::steady_clock::now();
  long time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();



  if(const char* path=getenv("HECBENCH_CCSD_TRACE")) {
    FILE* fp=fopen(path,"ab");if(!fp)throw std::runtime_error("trace open failed");
    const double values[]={emp4i,emp5i,emp4k,emp5k};
    if(fwrite(values,sizeof(double),4,fp)!=4)throw std::runtime_error("trace write failed");
    fclose(fp);
  }
  *emp4i_ = emp4i;
  *emp4k_ = emp4k;
  *emp5i_ = emp5i;
  *emp5k_ = emp5k;
  return time;
}
