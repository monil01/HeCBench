#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iris/iris.hpp>
#include <vector>
#include "utils.h"


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  int backend=0,type=0;char name[256]={};
  iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_type,&type,nullptr);
  iris_device_info(0,iris_name,name,nullptr);
  printf("Device backend=%d type=%d name=%s\n",backend,type,name);
  if(backend!=3 || type!=256) return 1;
  const int repeat = atoi(argv[1]);
  float *th, *pii, *q;
  float *qc, *qi, *qr, *qs;
  float *den, *p, *delz;
  float *rain,*rainncv;
  float *sr;
  float *snow, *snowncv;

  float delt = 10.f;
  int ims = 0, ime = 59, jms = 0, jme = 45, kms = 0, kme = 2;
  int ips = 0, ipe = 59, jps = 0, jpe = 45, kps = 0, kpe = 2;
  int d3 = (ime-ims+1) * (jme-jms+1) * (kme-kms+1) ;
  int d2 = (ime-ims+1) * (jme-jms+1) ;

  int dips = 0 ; int dipe = (ipe-ips+1) ;
  int djps = 0 ; int djpe = (jpe-jps+1) ;
  int dkps = 0 ; int dkpe = (kpe-kps+1) ;

  int remx = (ipe-ips+1) % XXX != 0 ? 1 : 0 ;
  int remy = (jpe-jps+1) % YYY != 0 ? 1 : 0 ;

  size_t lws[]={XXX,YYY};
  size_t gws[]={size_t((ipe-ips+1)/XXX+remx)*XXX,size_t((jpe-jps+1)/YYY+remy)*YYY};

  float rain_sum = 0, snow_sum = 0;

  long time = 0;
  for (int i = 0; i < repeat; i++) {
    // read-only in the kernel
    TODEV3(pii) ;
    TODEV3(den) ;
    TODEV3(p) ;
    TODEV3(delz) ;

    TODEV3(th) ;
    TODEV3(q) ;
    TODEV3(qc) ;
    TODEV3(qi) ;
    TODEV3(qr) ;
    TODEV3(qs) ;
    TODEV2(rain) ;
    TODEV2(rainncv) ;
    TODEV2(sr) ;
    TODEV2(snow) ;
    TODEV2(snowncv) ;

    auto start = std::chrono::steady_clock::now();

    int bounds[]={dips+1,ipe-ips+1,djps+1,jpe-jps+1,dkps+1,kpe-kps+1,
                  dips+1,dipe,djps+1,djpe,dkps+1,dkpe,dips+1,dipe,djps+1,djpe,dkps+1,dkpe};
    std::vector<void*> args={&th_d,&pii_d,&q_d,&qc_d,&qi_d,&qr_d,&qs_d,&den_d,&p_d,&delz_d,
                              &rain_d,&rainncv_d,&sr_d,&snow_d,&snowncv_d,&delt};
    std::vector<int> info={iris_rw,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,
                          iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(delt)};
    for(int& v:bounds) {args.push_back(&v);info.push_back(sizeof(v));}
    iris::Task task;
    if(task.kernel("wsm",2,nullptr,gws,lws,args.size(),args.data(),info.data())!=IRIS_SUCCESS) return 1;
    if(task.flush_out(th_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(q_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(qc_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(qi_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(qr_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(qs_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(rain_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(rainncv_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(sr_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(snow_d)!=IRIS_SUCCESS) return 1;
    if(task.flush_out(snowncv_d)!=IRIS_SUCCESS) return 1;
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 1;

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    FROMDEV2(rain) ;
    FROMDEV2(snow) ;

    rain_sum = snow_sum = 0;
    for (int i = 0; i < d2; i++) {
      rain_sum += rain[i];
      snow_sum += snow[i];
    }

    if(const char* path=getenv("WSM5_DUMP")) {
      FILE* f=fopen(path,"ab");if(!f) return 1;
      fwrite(th,sizeof(float),d3,f);
      fwrite(q,sizeof(float),d3,f);
      fwrite(qc,sizeof(float),d3,f);
      fwrite(qi,sizeof(float),d3,f);
      fwrite(qr,sizeof(float),d3,f);
      fwrite(qs,sizeof(float),d3,f);
      fwrite(rain,sizeof(float),d2,f);
      fwrite(rainncv,sizeof(float),d2,f);
      fwrite(sr,sizeof(float),d2,f);
      fwrite(snow,sizeof(float),d2,f);
      fwrite(snowncv,sizeof(float),d2,f);
      fclose(f);
    }
    FREE(pii) ;
    FREE(den) ;
    FREE(p) ;
    FREE(delz) ;
    FREE(th) ;
    FREE(q) ;
    FREE(qc) ;
    FREE(qi) ;
    FREE(qr) ;
    FREE(qs) ;
    FREE(rain) ;
    FREE(rainncv) ;
    FREE(sr) ;
    FREE(snow) ;
    FREE(snowncv) ;
  }

  printf("Average kernel execution time: %lf (ms)\n", (time * 1e-6) / repeat);
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);

  platform.finalize();
  return(0) ;
}
