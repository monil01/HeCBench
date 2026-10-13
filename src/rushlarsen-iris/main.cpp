#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include "dmem.h"
#include "utils.h"


int main(int argc, char *argv[])
{
  iris::Platform platform;
  if(platform.init(&argc,&argv,true)!=IRIS_SUCCESS) return 1;
  int dev=require_amd_hip(); if(dev<0) return 1;
  iris_device_set_default(dev);
  double t_start = 0;
  double dt = 0.02E-3;

  int num_timesteps = 1000000;
  int num_nodes = 1; 

  if(argc==2) { fprintf(stderr,"Need both timestep and node counts\n"); return 1; }
  if (argc > 1) {
    num_timesteps = atoi(argv[1]);
    printf("num_timesteps set to %d\n", num_timesteps);

    num_nodes = atoi(argv[2]);
    printf("num_nodes set to %d\n", num_nodes);

    if(num_timesteps <= 0 || num_nodes <= 0)
      exit(EXIT_FAILURE);
  }

  unsigned int num_states = NUM_STATES;
  size_t total_num_states = num_nodes * num_states;
  size_t states_size = total_num_states * sizeof(double);
  double *states = (double*) malloc(states_size);
  init_state_values(states, num_nodes);

  double *states2 = (double*) malloc(states_size);
  memcpy(states2, states, states_size);

  unsigned int num_parameters = NUM_PARAMS;
  size_t parameters_size = num_nodes * num_parameters * sizeof(double);
  double *parameters = (double*) malloc(parameters_size);
  init_parameters_values(parameters, num_nodes);

  double t = t_start;

  struct timespec timestamp_start, timestamp_now;
  double time_elapsed;

  printf("Host: Rush Larsen (exp integrator on all gates)\n");
  for (int it = 0; it < num_timesteps; it++) {
    forward_rush_larsen(states, t, dt, parameters, num_nodes);
    t += dt;
  }

  printf("Device: Rush Larsen (exp integrator on all gates)\n");
  iris::DMem d_states(states2, states_size);
  iris::DMem d_parameters(parameters, parameters_size);
  size_t grid[] = {static_cast<size_t>((num_nodes+255)/256)*256};
  size_t block[] = {256};

  t = t_start;


  clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_start);

  for (int it = 0; it < num_timesteps; it++) {
    iris::Task task;
    void* p[]={&d_states,&t,&dt,&d_parameters,&num_nodes};
    int info[]={iris_rw,sizeof(t),sizeof(dt),iris_r,sizeof(num_nodes)};
    if(task.kernel("k_forward_rush_larsen",1,nullptr,grid,block,5,p,info)!=IRIS_SUCCESS ||
       task.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS) return 1;  // run with a single node 
    t += dt;
  }


  clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp_now);
  time_elapsed = timestamp_now.tv_sec - timestamp_start.tv_sec + 1E-9 * (timestamp_now.tv_nsec - timestamp_start.tv_nsec);
  printf("Device: computed %d time steps in %g s. Time steps per second: %g\n\n",
      num_timesteps, time_elapsed, num_timesteps/time_elapsed);

  iris::Task flush; flush.flush_out(d_states);
  if(flush.submit(iris_gpu,nullptr,true)!=IRIS_SUCCESS) return 1;

  double rmse = 0.0;
  for (size_t i = 0; i < total_num_states; i++) {
    rmse += (states2[i] - states[i]) * (states2[i] - states[i]);
#ifdef VERBOSE
    printf("state[%d] = %lf\n", i, states[i]);
#endif
  }
  printf("RMSE = %lf\n", sqrt(rmse / (total_num_states)));
 
  if(const char* path=getenv("HECBENCH_DUMP")) {
    FILE* f=fopen(path,"wb"); if(!f) return 1;
    if(fwrite(states2,sizeof(double),total_num_states,f)!=total_num_states) return 1;
    fclose(f);
  }
  bool pass=true;
  for(size_t i=0;i<total_num_states;++i) {
    if(!std::isfinite(states2[i]) || !std::isfinite(states[i]) ||
       std::fabs(states2[i]-states[i])>1e-9+1e-9*std::fabs(states[i])) pass=false;
  }
  printf("Full state reference: %s\n",pass?"PASS":"FAIL");
  int errors=platform.error_count();
  free(states);
  free(states2);
  free(parameters);



  platform.finalize();
  return pass && !errors ? 0 : 1;
}
