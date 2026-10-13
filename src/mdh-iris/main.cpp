/*
 * A set of simple Multiple Debye-Huckel (MDH) kernels 
 * inspired by APBS:
 *   http://www.poissonboltzmann.org/ 
 *
 * This code was all originally written by David Gohara on MacOS X, 
 * and has been subsequently been modified by John Stone, porting to Linux,
 * adding vectorization, and several other platform-specific 
 * performance optimizations.
 * 
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "dmem.h"
#include "WKFUtils.h"


#define SEP printf("\n")

void gendata(float *ax,float *ay,float *az,
    float *gx,float *gy,float *gz,
    float *charge,float *size,int natom,int ngrid) {

  int i;    

  printf("Generating Data.. \n");
  for (i=0; i<natom; i++) {
    ax[i] = ((float) rand() / (float) RAND_MAX);
    ay[i] = ((float) rand() / (float) RAND_MAX);
    az[i] = ((float) rand() / (float) RAND_MAX);
    charge[i] = ((float) rand() / (float) RAND_MAX);
    size[i] = (float)natom;
  }

  for (i=0; i<ngrid; i++) {
    gx[i] = ((float) rand() / (float) RAND_MAX);
    gy[i] = ((float) rand() / (float) RAND_MAX);
    gz[i] = ((float) rand() / (float) RAND_MAX);
  }
  printf("Done generating inputs.\n\n");
}

void compare(const float *arr, const float *arr2, int ngrid) {
  int i;
  bool ok = true;
  for (i=0; i<ngrid; i++) {
    if (fabsf(arr[i] - arr2[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

void run_gpu_kernel(
    const int choice, 
    const int wgsize, 
    const int itmax,
    const int ngrid,
    const int natom,
    const int ngadj,
    const float *ax, 
    const float *ay,
    const float *az,
    const float *gx, 
    const float *gy, 
    const float *gz,
    const float *charge, 
    const float *size, 
    const float xkappa, 
    const float pre1, 
          float *val)
{
  wkf_timerhandle timer = wkf_timer_create();

  std::vector<float> val_host(ngadj,0.f);
  //Allocate memory for programs and kernels
  iris::DMem d_ax(const_cast<float*>(ax), sizeof(float)*natom);

  iris::DMem d_ay(const_cast<float*>(ay), sizeof(float)*natom);

  iris::DMem d_az(const_cast<float*>(az), sizeof(float)*natom);

  iris::DMem d_charge(const_cast<float*>(charge), sizeof(float)*natom);

  iris::DMem d_size(const_cast<float*>(size), sizeof(float)*natom);

  iris::DMem d_gx(const_cast<float*>(gx), sizeof(float)*ngadj);

  iris::DMem d_gy(const_cast<float*>(gy), sizeof(float)*ngadj);

  iris::DMem d_gz(const_cast<float*>(gz), sizeof(float)*ngadj);

  iris::DMem d_val(val_host.data(), sizeof(float)*ngadj);

  wkf_timer_start(timer); 

  // set work-item dimensions 
  // scale number of work units by vector size
  size_t grids[]={static_cast<size_t>(ngadj/4/wgsize)*wgsize};
  size_t blocks[]={static_cast<size_t>(wgsize)};


  for(int n = 0; n < itmax; n++) {
    iris::Task task;
    void* p[]={&d_ax,&d_ay,&d_az,&d_gx,&d_gy,&d_gz,&d_charge,&d_size,&d_val,const_cast<float*>(&pre1),const_cast<float*>(&xkappa),const_cast<int*>(&natom)};
    int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,sizeof(pre1),sizeof(xkappa),sizeof(natom)};
    const char* name=choice==0?"mdh":choice==1?"mdh2":"mdh3";
    if(task.kernel(name,1,nullptr,grids,blocks,12,p,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS) throw std::runtime_error(name);
  }


  wkf_timer_stop(timer);
  double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
  printf("Average kernel execution time: %1.12g\n", avg_kernel_time);

  // read output image
  iris::Task flush;flush.flush_out(d_val);
  if(flush.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS) throw std::runtime_error("flush");
  std::memcpy(val,val_host.data(),sizeof(float)*ngrid);











  wkf_timer_destroy(timer);
}

// reference CPU kernel
void run_cpu_kernel(
    const int itmax,
    const int ngrid,
    const int natom,
    const float *ax,
    const float *ay,
    const float *az,
    const float *gx,
    const float *gy,
    const float *gz,
    const float *charge,
    const float *size,
    const float xkappa,
    const float pre1,
          float *val)
{
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer); 

  for(int n = 0; n < itmax; n++) {
    #pragma omp parallel for
    for(int igrid=0;igrid<ngrid;igrid++){
      float sum = 0.0f;
      #pragma omp parallel for simd reduction(+:sum)
      for(int iatom=0; iatom<natom; iatom++) {
        float dist = sqrtf((gx[igrid]-ax[iatom])*(gx[igrid]-ax[iatom]) + 
            (gy[igrid]-ay[iatom])*(gy[igrid]-ay[iatom]) + 
            (gz[igrid]-az[iatom])*(gz[igrid]-az[iatom]));

        sum += pre1*(charge[iatom]/dist)*expf(-xkappa*(dist-size[iatom]))
          / (1+xkappa*size[iatom]);
      }
      val[igrid] = sum;
    }
  }

  wkf_timer_stop(timer);
  double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
  printf("Average kernel execution time: %1.12g\n", avg_kernel_time);

  wkf_timer_destroy(timer);
}


void usage() {
  printf("command line parameters:\n");
  printf("Optional test flags:\n");
  printf("  -itmax N         loop test N times\n");
  printf("  -wgsize          set workgroup size\n");
}

void getargs(int argc, const char **argv, int *itmax, int *wgsize) {
  int i;
  for (i=0; i<argc; i++) {
    if ((!strcmp(argv[i], "-itmax")) && ((i+1) < argc)) {
      i++;
      *itmax = atoi(argv[i]);
    }

    if ((!strcmp(argv[i], "-wgsize")) && ((i+1) < argc)) {
      i++;
      *wgsize = atoi(argv[i]);
    }
  }

  printf("Run parameters:\n");
  printf("  kernel loop count: %d\n", *itmax);
  printf("     workgroup size: %d\n", *wgsize);
}


int main(int argc, const char **argv) {
  iris::Platform platform;
  char** mutable_argv=const_cast<char**>(argv);
  if(platform.init(&argc,&mutable_argv,true)!=IRIS_SUCCESS)return 1;
  int dev=require_amd_hip();if(dev<0)return 1;iris_device_set_default(dev);
  srand(1);
  int itmax = 100;
  int wgsize = 256;

  getargs(argc, argv, &itmax, &wgsize);

  wkf_timerhandle timer = wkf_timer_create();

  int natom = 5877;
  int ngrid = 134918;
  int ngadj = ngrid + (512 - (ngrid & 511));

  float pre1 = 4.46184985145e19;
  float xkappa = 0.0735516324639;
#ifdef HECBENCH_MDH_VALIDATION
  natom=37; ngrid=1030; ngadj=ngrid+(512-(ngrid&511));pre1=1.f;
#endif

  float *ax = (float*)calloc(natom, sizeof(float));
  float *ay = (float*)calloc(natom, sizeof(float));
  float *az = (float*)calloc(natom, sizeof(float));
  float *charge = (float*)calloc(natom, sizeof(float));
  float *size = (float*)calloc(natom, sizeof(float));

  float *gx = (float*)calloc(ngadj, sizeof(float));
  float *gy = (float*)calloc(ngadj, sizeof(float));
  float *gz = (float*)calloc(ngadj, sizeof(float));

  // result
  float *val_cpu = (float*)calloc(ngadj, sizeof(float));
  float *val_gpu = (float*)calloc(ngadj, sizeof(float));

  gendata(ax, ay, az, gx, gy, gz, charge, size, natom, ngrid);

  wkf_timer_start(timer);
  run_cpu_kernel(itmax, ngadj, natom, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val_cpu);
  wkf_timer_stop(timer);

  printf("CPU Time: %1.12g (Number of tests = %d)\n", wkf_timer_time(timer), itmax);
  SEP;

  for (int choice = 0; choice < 3; choice++) {
    wkf_timer_start(timer);
    run_gpu_kernel(choice, wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, 
                   charge, size, xkappa, pre1, val_gpu);
    wkf_timer_stop(timer);

    printf("GPU Time: %1.12g (Number of tests = %d)\n", wkf_timer_time(timer), itmax);
    SEP;
    if(const char* prefix=getenv("HECBENCH_DUMP")) {
      char path[512];snprintf(path,sizeof(path),"%s.%d",prefix,choice);
      FILE* f=fopen(path,"wb");if(!f)return 1;
      if(fwrite(val_gpu,sizeof(float),ngrid,f)!=static_cast<size_t>(ngrid))return 1;
      fclose(f);
    }
    compare(val_cpu, val_gpu, ngrid);
  }

  free(ax);
  free(ay);
  free(az);
  free(charge);
  free(size);
  free(gx);
  free(gy);
  free(gz);
  free(val_cpu);
  free(val_gpu);

  wkf_timer_destroy(timer);

  int errors=platform.error_count();platform.finalize();return errors?1:0;
}
