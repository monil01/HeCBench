#include <iris/iris.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
extern "C" void offload(double* h_flux_0,double* h_flux_1,double* h_flux_2,double* h_flux_3,double* h_flux_4,double* h_cons_1,double* h_cons_2,double* h_cons_3,double* h_cons_4,double* h_q_1,double* h_q_2,double* h_q_3,double* h_q_4,double dxinv0,double dxinv1,double dxinv2,int L,int M,int N,int repeat) {
  iris::Platform platform;
  if(platform.init(nullptr,nullptr,1)!=IRIS_SUCCESS) exit(1);
  bool gpu=false;
  for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }
  if(!gpu) exit(1);
  size_t bytes=sizeof(double)*L*M*N;
  iris::DMem flux_0(h_flux_0,bytes);
  iris::DMem flux_1(h_flux_1,bytes);
  iris::DMem flux_2(h_flux_2,bytes);
  iris::DMem flux_3(h_flux_3,bytes);
  iris::DMem flux_4(h_flux_4,bytes);
  iris::DMem cons_1(h_cons_1,bytes);
  iris::DMem cons_2(h_cons_2,bytes);
  iris::DMem cons_3(h_cons_3,bytes);
  iris::DMem cons_4(h_cons_4,bytes);
  iris::DMem q_1(h_q_1,bytes);
  iris::DMem q_2(h_q_2,bytes);
  iris::DMem q_3(h_q_3,bytes);
  iris::DMem q_4(h_q_4,bytes);
  size_t gws[]={size_t((N+15)/16)*16,size_t((M+3)/4)*4,size_t((L+3)/4)*4},lws[]={16,4,4};
  long times[3]={};
  for(int r=0;r<repeat;r++) {
    flux_0.update(h_flux_0); flux_1.update(h_flux_1); flux_2.update(h_flux_2); flux_3.update(h_flux_3); flux_4.update(h_flux_4);
    for(int k=0;k<3;k++) {
      iris::Task task;
      void* args[]={&flux_0,&flux_1,&flux_2,&flux_3,&flux_4,&cons_1,&cons_2,&cons_3,&cons_4,&q_1,&q_2,&q_3,&q_4,&dxinv0,&dxinv1,&dxinv2,&L,&M,&N};
      int info[]={iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,sizeof(dxinv0),sizeof(dxinv1),sizeof(dxinv2),sizeof(L),sizeof(M),sizeof(N)};
      char name[32];snprintf(name,sizeof(name),"hypterm_%d",k+1);
      auto start=std::chrono::steady_clock::now();
      if(task.kernel(name,3,nullptr,gws,lws,19,args,info)!=IRIS_SUCCESS) exit(2);
      if(r==repeat-1 && k==2) { task.flush_out(flux_0); task.flush_out(flux_1); task.flush_out(flux_2); task.flush_out(flux_3); task.flush_out(flux_4); }
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      times[k]+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    }
  }
  for(int k=0;k<3;k++)printf("Average kernel execution time (k%d): %f (ms)\n",k+1,times[k]*1e-6/repeat);
  if(const char* path=getenv("HECBENCH_ARTIFACT")) {
    FILE* out=fopen(path,"wb");if(!out)exit(3);
    fwrite(h_flux_0,1,bytes,out); fwrite(h_flux_1,1,bytes,out); fwrite(h_flux_2,1,bytes,out); fwrite(h_flux_3,1,bytes,out); fwrite(h_flux_4,1,bytes,out); fclose(out);
  }
  int errors=platform.error_count();platform.finalize();if(errors)exit(2);
}
