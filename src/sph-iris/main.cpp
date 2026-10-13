#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"

////////////////////////////////////////////////////////////////////////////
// B spline smoothing kernel
////////////////////////////////////////////////////////////////////////////

#include <iris/iris.hpp>
// Seed simulation with Euler step v(t-dt/2) needed by leap frog integrator
// Should calculate all accelerations but assuming just g simplifies acc port
void eulerStart(fluid_particle* fluid_particles,
                boundary_particle* boundary_particles, param *params)
{
    // Set V (t0 - dt/2)
    double dt_half = params->time_step/2.0;

    for(int i=0; i<params->number_fluid_particles; i++)
    {
        // Velocity at t + dt/2
        double3 v      = fluid_particles[i].v;
        double3 v_half;

        v_half.x = v.x;
        v_half.y = v.y;
        v_half.z = v.z - params->g * dt_half;

        fluid_particles[i].v_half = v_half;
    }
}

// Initialize particles
void initParticles(fluid_particle** fluid_particles, boundary_particle** boundary_particles,
                   AABB* water, AABB* boundary, param* params)
{
    // Allocate fluid particles array
    *fluid_particles = (fluid_particle*) malloc(params->number_fluid_particles * sizeof(fluid_particle));
    // Allocate boundary particles array
    *boundary_particles = (boundary_particle*) malloc(params->number_boundary_particles * sizeof(boundary_particle));

    double spacing = params->spacing_particle;

    // Initialize particle values
    for(int i=0; i<params->number_fluid_particles; i++) {
        (*fluid_particles)[i].a.x = 0.0;
        (*fluid_particles)[i].a.y = 0.0;
        (*fluid_particles)[i].a.z = 0.0;
        (*fluid_particles)[i].v.x = 0.0;
        (*fluid_particles)[i].v.y = 0.0;
        (*fluid_particles)[i].v.z = 0.0;
        (*fluid_particles)[i].density = params->rest_density;
    }

    // Place particles inside bounding volume
    double x,y,z;
    int i = 0;
    for(z=water->min_z; z<=water->max_z; z+=spacing) {
        for(y=water->min_y; y<=water->max_y; y+=spacing) {
            for(x=water->min_x; x<=water->max_x; x+=spacing) {
                if(i < params->number_fluid_particles) {
                    (*fluid_particles)[i].pos.x = x;
                    (*fluid_particles)[i].pos.y = y;
                    (*fluid_particles)[i].pos.z = z;
                    i++;
                }
            }
        }
    }
    params->number_fluid_particles = i;

    // Construct bounding box
    constructBoundaryBox(*boundary_particles, boundary, params);
}

void initParams(AABB* water_volume, AABB* boundary_volume, param* params)
{
    // Boundary box
    boundary_volume->min_x = 0.0;
    boundary_volume->max_x = 1.1;
    boundary_volume->min_y = 0.0;
    boundary_volume->max_y = 1.1;
    boundary_volume->min_z = 0.0;
    boundary_volume->max_z = 1.1;

    // water volume
    water_volume->min_x = 0.1;
    water_volume->max_x = 0.5;
    water_volume->min_y = 0.1;
    water_volume->max_y = 0.5;
    water_volume->min_z = 0.08;
    water_volume->max_z = 0.8;

    // Simulation parameters
    params->number_fluid_particles = 2048;
    params->rest_density = 1000.0;
    params->g = 9.8;
    params->alpha = 0.02;
    params->surface_tension =  0.01;
#ifdef VALIDATION_STEPS
    params->number_steps = VALIDATION_STEPS;
#else
    params->number_steps = 500; // reduce from 5000
#endif
    params->time_step = 0.00035;

    // Mass of each particle
    double volume = (water_volume->max_x - water_volume->min_x)
                  * (water_volume->max_y - water_volume->min_y)
                  * (water_volume->max_z - water_volume->min_z);
    params->mass_particle = params->rest_density * (volume/params->number_fluid_particles);

    // Cube calculated spacing
    params->spacing_particle = pow(volume/params->number_fluid_particles,1.0/3.0);

    // Smoothing radius, h
    params->smoothing_radius = params->spacing_particle;

    // Boundary particles
    int num_x = ceil((boundary_volume->max_x - boundary_volume->min_x)/params->spacing_particle);
    int num_y = ceil((boundary_volume->max_y - boundary_volume->min_y)/params->spacing_particle);
    int num_z = ceil((boundary_volume->max_z - boundary_volume->min_z)/params->spacing_particle);
    int num_boundary_particles = (2 * num_x * num_z) + (2 * num_y * num_z) + (2* num_y * num_z);
    params->number_boundary_particles = num_boundary_particles;

    // Total number of particles
    params->number_particles = params->number_boundary_particles + params->number_fluid_particles;

    // Number of steps before frame needs to be written for 30 fps
    params->steps_per_frame = (int)(1.0/(params->time_step*30.0));

    // Calculate speed of sound for simulation
    double max_height = water_volume->max_y;
    double max_velocity = sqrt(2.0*params->g*max_height);
    params->speed_sound = max_velocity/sqrt(0.01);

    // Minimum stepsize from Courant-Friedrichs-Lewy condition
    double recomend_step = 0.4 * params->smoothing_radius / (params->speed_sound
                                                         * (1+ 0.6*params->alpha));
    printf("Using time step: %f, Minimum recomended %f\n",params->time_step, recomend_step);
}

void finalizeParticles(fluid_particle *fluid_particles, boundary_particle *boundary_particles)
{
    free(fluid_particles);
    free(boundary_particles);
}

int main(int argc, char *argv[])
{
    iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS)return 1;
    bool gpu=false;for(int d=0;d<platform.ndevs();d++) {
      int backend=0,type=0;char name[256]={};size_t bytes=0;
      iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
      printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
    }if(!gpu)return 1;
    param params;
    AABB water_volume;
    AABB boundary_volume;
    fluid_particle *fluid_particles = NULL;
    boundary_particle *boundary_particles = NULL;
    initParams(&water_volume, &boundary_volume, &params);

    initParticles(&fluid_particles, &boundary_particles, &water_volume,
                  &boundary_volume, &params);

    eulerStart(fluid_particles, boundary_particles, &params);

    int num_fluid_particles = params.number_fluid_particles;
    int num_boundary_particles = params.number_boundary_particles;

    iris::DMem d_fluid_particles(fluid_particles,num_fluid_particles*sizeof(fluid_particle));
    iris::DMem d_boundary_particles(boundary_particles,num_boundary_particles*sizeof(boundary_particle));
    iris::DMem d_params(&params,sizeof(param));

    dim3 block1D(256);
    dim3 grid1D_FP((num_fluid_particles + 255)/256);
    dim3 grid1D_BP((num_boundary_particles + 255)/256);

    auto start = std::chrono::steady_clock::now();

    // Main simulation loop
    for(int n=0; n<params.number_steps; n++) {
        {
          iris::Task task;size_t gws[]={size_t(grid1D_FP.x)*block1D.x},lws[]={block1D.x};
          void* args[]={&d_fluid_particles,&d_params};int info[]={iris_rw,iris_r};
          if(task.kernel("updatePressures",1,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS)return 2;
          if(n==params.number_steps-1 && std::string("updatePressures")=="updatePositions") task.flush_out(d_fluid_particles);
          if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
        }
        {
          iris::Task task;size_t gws[]={size_t(grid1D_FP.x)*block1D.x},lws[]={block1D.x};
          void* args[]={&d_fluid_particles,&d_params};int info[]={iris_rw,iris_r};
          if(task.kernel("updateAccelerationsFP",1,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS)return 2;
          if(n==params.number_steps-1 && std::string("updateAccelerationsFP")=="updatePositions") task.flush_out(d_fluid_particles);
          if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
        }
        {
          iris::Task task;size_t gws[]={size_t(grid1D_BP.x)*block1D.x},lws[]={block1D.x};
          void* args[]={&d_fluid_particles,&d_boundary_particles,&d_params};int info[]={iris_rw,iris_r,iris_r};
          if(task.kernel("updateAccelerationsBP",1,nullptr,gws,lws,3,args,info)!=IRIS_SUCCESS)return 2;
          if(n==params.number_steps-1 && std::string("updateAccelerationsBP")=="updatePositions") task.flush_out(d_fluid_particles);
          if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
        }
        {
          iris::Task task;size_t gws[]={size_t(grid1D_FP.x)*block1D.x},lws[]={block1D.x};
          void* args[]={&d_fluid_particles,&d_params};int info[]={iris_rw,iris_r};
          if(task.kernel("updatePositions",1,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS)return 2;
          if(n==params.number_steps-1 && std::string("updatePositions")=="updatePositions") task.flush_out(d_fluid_particles);
          if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of sph kernels: %f (ms)\n", (time * 1e-6f) / params.number_steps);



    if(const char* path=getenv("HECBENCH_ARTIFACT")) {
      FILE* out=fopen(path,"wb");if(!out)return 3;
      fwrite(&num_fluid_particles,sizeof(int),1,out);
      for(int i=0;i<num_fluid_particles;i++) {
        const auto& p=fluid_particles[i];
        double fields[]={p.density,p.pressure,p.pos.x,p.pos.y,p.pos.z,p.v.x,p.v.y,p.v.z,p.v_half.x,p.v_half.y,p.v_half.z,p.a.x,p.a.y,p.a.z};
        fwrite(fields,sizeof(double),14,out);
      }fclose(out);
    }
    writeFile(fluid_particles, &params);

    finalizeParticles(fluid_particles, boundary_particles);



    int errors=platform.error_count();platform.finalize();return errors ? 1 : 0;
}
