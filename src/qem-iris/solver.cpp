#include <iris/iris.hpp>
#include <vector>
#include <stdexcept>
#include "gpu_solver.h"
extern int gpu_device;
void QuarticMinimumGPU(int N,float* A,float* B,float* C,float* D,float* E,float* minimum) {
  (void)E;
  const size_t bytes=N*sizeof(float);
  iris::DMem a(A,bytes),b(B,bytes),c(C,bytes),d(D,bytes),out(minimum,bytes);
  std::vector<float> hb(N),hc(N),hd(N),hq(N),hr(N),hqi(N),hri(N),hdel(N);
  std::vector<float> ht(N),hs(N),hx1(N),hx2(N),hx3(N),htemp(N);
  iris::DMem bi(hb.data(),bytes),ci(hc.data(),bytes),di(hd.data(),bytes),q(hq.data(),bytes),r(hr.data(),bytes);
  iris::DMem qi(hqi.data(),bytes),ri(hri.data(),bytes),del(hdel.data(),bytes),theta(ht.data(),bytes),sqrtq(hs.data(),bytes);
  iris::DMem x1(hx1.data(),bytes),x2(hx2.data(),bytes),x3(hx3.data(),bytes),temp(htemp.data(),bytes);
  size_t global[]={size_t((N+63)/64)*64},local[]={64};
  {
    iris::Task task;
    void* p[]={&N,&a,&b,&c,&d,&bi,&ci,&di,&q,&r,&qi,&ri,&del};
    int info[]={sizeof(N),iris_r,iris_r,iris_r,iris_r,iris_w,iris_w,iris_w,iris_w,iris_w,iris_w,iris_w,iris_w};
    if(task.kernel("QRdel",1,nullptr,global,local,13,p,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("QRdel failed");
  }
  {
    iris::Task task;
    void* p[]={&N,&a,&b,&c,&d,&bi,&q,&r,&del,&theta,&sqrtq,&x1,&x2,&x3,&temp,&out};
    int info[]={sizeof(N),iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,iris_w,iris_w,iris_w,iris_w,iris_rw,iris_w};
    if(task.kernel("QuarticSolver",1,nullptr,global,local,16,p,info)) throw std::runtime_error("Quartic setup failed");
    task.flush_out(out);
    if(task.submit(gpu_device,nullptr,1)) throw std::runtime_error("Quartic failed");
  }
}
void QuarticMinimumGPUStreams(int N,float* A,float* B,float* C,float* D,float* E,float* minimum) {
  // Preserve the four original chunks and residual coverage using synchronous IRIS tasks.
  const int chunk=N/4;
  for(int i=0;i<4;++i) if(chunk) QuarticMinimumGPU(chunk,A+i*chunk,B+i*chunk,C+i*chunk,D+i*chunk,E+i*chunk,minimum+i*chunk);
  const int remainder=N%4;
  if(remainder) QuarticMinimumGPU(remainder,A+4*chunk,B+4*chunk,C+4*chunk,D+4*chunk,E+4*chunk,minimum+4*chunk);
}
