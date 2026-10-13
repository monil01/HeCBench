#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <iris/iris.hpp>
#include <cstdlib>
#include <stdexcept>
constexpr int L_max = 64;

const double t_init = .1;
const double t_final = 200;

void initial(double c[], int seed)
{
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dis(-6.0, 6.0);
  std::uniform_real_distribution<double> temp(0.1, 0.9);
  for(int l = 1; l < L_max; l++) c[l] = dis(gen);
  c[0] = temp(gen) ; // random temperature
  c[L_max] = 0.0; //truncation; do not modify
}

#ifdef DUMP
void dump (FILE * out, double c[], double t)
{
  fprintf(out,"%.5e ", t);
  int L = (L_max > 4) ? 4 : L_max; // print a subset 
  for(int l = 0; l < L; l++)
  {
    fprintf(out,"%.5e ", c[l]);
  }
  fprintf(out,"\n");
  fflush(out);
}
#endif

int main (int argc, char* argv[]) {

#ifdef DUMP
  FILE * outdata = fopen ("output.csv","w");
  if (outdata == NULL) {
    printf("Failed to open file for write\n");
    return 1;
  }
#endif

  double tmin = t_init;
  double tmax = t_final;
  double delta_t = 0.1;

  int dimension = L_max;
  int seed = dimension;

  int size = sizeof(double) * (dimension+1);

  double *c = (double*) malloc (size);
  double *n = (double*) malloc (size);

  initial(c, seed);
#ifdef DUMP
  dump(outdata, c, tmin);
#endif

  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  iris::DMem d_c(c,size), d_n(n,size);
  FILE* full_output=nullptr;
  if(const char* path=getenv("HECBENCH_RHS_OUTPUT")) {
    full_output=fopen(path,"wb"); if(!full_output) return 2;
  }

  float total_time = 0.f;
  for (double t_next = tmin + delta_t; t_next <= tmax; t_next += delta_t)
  {
    d_c.update(c); 

    auto start = std::chrono::steady_clock::now();

    iris::Task task;
    void* params[]={&t_next,&d_c,&d_n};
    int info[]={sizeof(t_next),iris_r,iris_rw};
    size_t global[]={96},local[]={96};
    if(task.kernel("RHS_f",1,nullptr,global,local,3,params,info)) return 2;
    task.flush_out(d_n);
    if(task.submit(device,nullptr,1)) return 2;
    if(full_output && fwrite(n,sizeof(double),L_max,full_output)!=L_max) return 2;

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

#ifdef DUMP
    dump(outdata, n, t_next);
#endif
    initial(c, ++seed);
  }
  printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);

#ifdef DUMP
  fclose (outdata);
#endif

  if(full_output) fclose(full_output);
  platform.finalize();
  free(c);
  free(n);

  return 0;
}
