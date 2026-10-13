/********************************************************************
  euler3d.cpp
  : parallelized code of CFD

  - original code from the AIAA-2009-4001 by Andrew Corrigan, acorriga@gmu.edu
  - parallelization with OpenCL API has been applied by
  Jianbin Fang - j.fang@tudelft.nl
  Delft University of Technology
  Faculty of Electrical Engineering, Mathematics and Computer Science
  Department of Software Technology
  Parallel and Distributed Systems Group
  on 24/03/2011
 ********************************************************************/

#include <iostream>
#include <fstream>
#include <math.h>
#include <iris/iris.hpp>
#include <vector>
#include <stdexcept>
#include "util.h"

/*
 * Options 
 * 
 */ 
#define GAMMA 1.4f
#define iterations 2000
#ifndef block_length
#define block_length 192
#endif

#define NDIM 3
#define NNB 4

#define RK 3  // 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif

//self-defined user type
typedef struct{
  float x;
  float y;
  float z;
} Float3;


inline void compute_velocity(const float density, const Float3 momentum, Float3* velocity){
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}


inline float compute_speed_sqd(const Float3 velocity){
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}


inline float compute_pressure(const float density, const float density_energy, const float speed_sqd){
  return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
// sqrt is a device function

inline float compute_speed_of_sound(const float density, const float pressure){
  return sqrt((float)(GAMMA)*pressure/density);
}
   
inline void compute_flux_contribution(const float density, 
    Float3 momentum, 
    const float density_energy, 
    const float pressure, 
    const Float3 velocity, 
    Float3* fc_momentum_x, 
    Float3* fc_momentum_y, 
    Float3* fc_momentum_z, 
    Float3* fc_density_energy)
{
  fc_momentum_x->x = velocity.x*momentum.x + pressure;
  fc_momentum_x->y = velocity.x*momentum.y;
  fc_momentum_x->z = velocity.x*momentum.z;


  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y*momentum.y + pressure;
  fc_momentum_y->z = velocity.y*momentum.z;

  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z*momentum.z + pressure;

  const float de_p = density_energy+pressure;
  fc_density_energy->x = velocity.x*de_p;
  fc_density_energy->y = velocity.y*de_p;
  fc_density_energy->z = velocity.z*de_p;
}


static int gpu_device=-1;
struct Param {void* ptr;int info;};
static Param param(iris::DMem& mem){return {&mem,iris_rw};}
template<class T> static Param param(const T& value){return {const_cast<T*>(&value),sizeof(T)};}
template<class... Args> static void launch(const char* name,size_t grid,size_t block,Args&&... args) {
  if(block<1 || block>1024) throw std::runtime_error("block size must be 1..1024");
  Param list[]={param(args)...};void* p[sizeof...(args)];int info[sizeof...(args)];
  for(size_t i=0;i<sizeof...(args);++i){p[i]=list[i].ptr;info[i]=list[i].info;}
  size_t g[]={grid*block},l[]={block};iris::Task task;
  if(task.kernel(name,1,nullptr,g,l,sizeof...(args),p,info)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("kernel failed");
}
static void fetch(iris::DMem& mem,void* host) {
  iris::Task task;task.flush_out(mem);
  if(task.submit(gpu_device,nullptr,1) || mem.fetch(host)) throw std::runtime_error("fetch failed");
}
void copy(iris::DMem& dst,iris::DMem& src,const int N) {
  (void)N; iris::Task task;
  if(task.dmem2dmem(&src,&dst)||task.submit(gpu_device,nullptr,1)) throw std::runtime_error("DMem copy failed");
}
void dump(const float *h_variables, const int nel, const int nelr){

  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
  }


  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++)
        file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
      file << std::endl;
    }
  }

  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
  }
}


/*
 * Main function
 */
int main(int argc, char** argv){
  printf("WG size of kernel:initialize = %d\nWG size of kernel:compute_step_factor = %d\nWG size of kernel:compute_flux = %d\nWG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);

  if (argc < 2){
    std::cout << "Please specify data file name" << std::endl;
    return 0;
  }
  const char* data_file_name = argv[1];
  float h_ff_variable[NVAR];

  std::ifstream file(data_file_name, std::ifstream::in);
  if(!file.good()){
    std::cerr << "Failed to find/open data file. Exit" << std::endl;
    return 1;
  }

  // set far field conditions and load them into constant memory on the gpu
  //{
  const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  h_ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach)*ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
  ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
  h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
  h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

  h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

  Float3 h_ff_momentum;
  h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
  h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
  h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
  Float3 h_ff_flux_contribution_momentum_x;
  Float3 h_ff_flux_contribution_momentum_y;
  Float3 h_ff_flux_contribution_momentum_z;
  Float3 h_ff_flux_contribution_density_energy;
  compute_flux_contribution(h_ff_variable[VAR_DENSITY], 
      h_ff_momentum, 
      h_ff_variable[VAR_DENSITY_ENERGY], 
      ff_pressure,
      ff_velocity, 
      &h_ff_flux_contribution_momentum_x, 
      &h_ff_flux_contribution_momentum_y, 
      &h_ff_flux_contribution_momentum_z,
      &h_ff_flux_contribution_density_energy);

  int nel;
  int nelr;
  file >> nel;
  nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));
  std::cout<<"--cambine: nel="<<nel<<", nelr="<<nelr<<std::endl;
  float* h_areas = new float[nelr];
  int* h_elements_surrounding_elements = new int[nelr*NNB];
  float* h_normals = new float[nelr*NDIM*NNB];


  // read in data
  for(int i = 0; i < nel; i++)
  {
    file >> h_areas[i];
    for(int j = 0; j < NNB; j++)
    {
      file >> h_elements_surrounding_elements[i + j*nelr];
      if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
      h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering        

      for(int k = 0; k < NDIM; k++)
      {
        file >> h_normals[i + (j + k*NNB)*nelr];
        h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
      }
    }
  }

  // fill in remaining data
  int last = nel-1;
  for(int i = nel; i < nelr; i++)
  {
    h_areas[i] = h_areas[last];
    for(int j = 0; j < NNB; j++)
    {
      // duplicate the last element
      h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];  
      for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
    }
  }

  float* h_variables = new float[nelr*NVAR];

#ifdef DEBUG
  float* h_step_factors = new float[nelr]; 
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
  gpu_device=device;
  auto offload_start = std::chrono::steady_clock::now();


  iris::DMem d_ff_variable(h_ff_variable,sizeof(float)*NVAR);
  iris::DMem d_ff_flux_contribution_momentum_x(&h_ff_flux_contribution_momentum_x,sizeof(Float3));
  iris::DMem d_ff_flux_contribution_momentum_y(&h_ff_flux_contribution_momentum_y,sizeof(Float3));
  iris::DMem d_ff_flux_contribution_momentum_z(&h_ff_flux_contribution_momentum_z,sizeof(Float3));
  iris::DMem d_ff_flux_contribution_density_energy(&h_ff_flux_contribution_density_energy,sizeof(Float3));

  iris::DMem d_areas(h_areas,sizeof(float)*nelr);

  iris::DMem d_normals(h_normals,sizeof(float)*nelr*NDIM*NNB);

  iris::DMem d_elements_surrounding_elements(h_elements_surrounding_elements,sizeof(int)*nelr*NNB);

  // Create arrays and set initial conditions
  iris::DMem d_variables(h_variables,sizeof(float)*nelr*NVAR);

  std::vector<float> d_old_variables_host((sizeof(float)*nelr*NVAR)/sizeof(float));
  iris::DMem d_old_variables(d_old_variables_host.data(),sizeof(float)*nelr*NVAR);

  std::vector<float> d_fluxes_host((sizeof(float)*nelr*NVAR)/sizeof(float));
  iris::DMem d_fluxes(d_fluxes_host.data(),sizeof(float)*nelr*NVAR);

  std::vector<float> d_step_factors_host((sizeof(float)*nelr)/sizeof(float));
  iris::DMem d_step_factors(d_step_factors_host.data(),sizeof(float)*nelr);

  size_t gridDim1 = (nelr + BLOCK_SIZE_1 - 1)/BLOCK_SIZE_1;
  size_t gridDim2 = (nelr + BLOCK_SIZE_2 - 1)/BLOCK_SIZE_2;
  size_t gridDim3 = (nelr + BLOCK_SIZE_3 - 1)/BLOCK_SIZE_3;
  size_t gridDim4 = (nelr + BLOCK_SIZE_4 - 1)/BLOCK_SIZE_4;

  auto kernel_start = std::chrono::steady_clock::now();

  launch("initialize_variables",gridDim1,BLOCK_SIZE_1,nelr, d_variables, d_ff_variable);
  launch("initialize_variables",gridDim1,BLOCK_SIZE_1,nelr, d_old_variables, d_ff_variable);  
  launch("initialize_variables",gridDim1,BLOCK_SIZE_1,nelr, d_fluxes, d_ff_variable);    
  launch("initialize_buffer",gridDim1,BLOCK_SIZE_1,d_step_factors, 0.0f, nelr);

  // Begin iterations
  for(int n = 0; n < (getenv("HECBENCH_CFD_ITERATIONS")?atoi(getenv("HECBENCH_CFD_ITERATIONS")):iterations); n++){
    copy(d_old_variables, d_variables, nelr*NVAR);

    // for the first iteration we compute the time step
    launch("compute_step_factor",gridDim2,BLOCK_SIZE_2,nelr, d_variables, d_areas, d_step_factors);

#ifdef DEBUG
    fetch(d_step_factors,h_step_factors);
    for (int i = 0; i < 16; i++) printf("step factor: i=%d %f\n", i, h_step_factors[i]);
#endif
    for(int j = 0; j < RK; j++){
      launch("compute_flux",gridDim3,BLOCK_SIZE_3,nelr, d_elements_surrounding_elements, d_normals, 
          d_variables, d_ff_variable, d_fluxes, d_ff_flux_contribution_density_energy, 
          d_ff_flux_contribution_momentum_x, d_ff_flux_contribution_momentum_y, 
          d_ff_flux_contribution_momentum_z);
      launch("time_step",gridDim4,BLOCK_SIZE_4,j, nelr, d_old_variables, d_variables, d_step_factors, d_fluxes);
    }
  }

  auto kernel_end = std::chrono::steady_clock::now();

  fetch(d_variables,h_variables);


  auto offload_end = std::chrono::steady_clock::now();
  auto offload_time = std::chrono::duration_cast<std::chrono::nanoseconds>(offload_end - offload_start).count();
  printf("Device offloading time = %lf(s)\n", offload_time * 1e-9);

  auto kernel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count();
  printf("Total execution time of kernels = %lf(s)\n", kernel_time * 1e-9);

#ifdef OUTPUT
  std::cout << "Saving solution..." << std::endl;
  dump(h_variables, nel, nelr);
#endif

  if(const char* path=getenv("HECBENCH_CFD_OUTPUT")) {
    FILE* fp=fopen(path,"wb");if(!fp)return 2;
    for(int var=0;var<NVAR;++var)
      if(fwrite(h_variables+var*nelr,sizeof(float),nel,fp)!=static_cast<size_t>(nel))return 2;
    fclose(fp);
  }
  platform.finalize();
  delete[] h_areas;
  delete[] h_elements_surrounding_elements;
  delete[] h_normals;
  delete[] h_variables;
#ifdef DEBUG
  delete[] h_step_factors;
#endif

  std::cout << "Done..." << std::endl;

  return 0;
}
