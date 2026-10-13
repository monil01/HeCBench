/*
 ** Code to implement a d2q9-bgk lattice boltzmann scheme.
 ** 'd2' inidates a 2-dimensional grid, and
 ** 'q9' indicates 9 velocities per grid cell.
 ** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 **
 ** The 'speeds' in each cell are numbered as follows:
 **
 ** 6 2 5
 **  \|/
 ** 3-0-1
 **  /|\
 ** 7 4 8
 **
 ** A 2D grid:
 **
 **           cols
 **       --- --- ---
 **      | D | E | F |
 ** rows  --- --- ---
 **      | A | B | C |
 **       --- --- ---
 **
 ** 'unwrapped' in row major order to give a 1D array:
 **
 **  --- --- --- --- --- ---
 ** | A | B | C | D | E | F |
 **  --- --- --- --- --- ---
 **
 ** Grid indicies are:
 **
 **          ny
 **          ^       cols(ii)
 **          |  ----- ----- -----
 **          | | ... | ... | etc |
 **          |  ----- ----- -----
 ** rows(jj) | | 1,0 | 1,1 | 1,2 |
 **          |  ----- ----- -----
 **          | | 0,0 | 0,1 | 0,2 |
 **          |  ----- ----- -----
 **          ----------------------> nx
 **
 ** Note the names of the input parameter and obstacle files
 ** are passed on the command line, e.g.:
 **
 **   ./d2q9-bgk input.params obstacles.dat
 **
 ** Be sure to adjust the grid dimensions in the parameter file
 ** if you choose a different obstacle file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <iris/iris.hpp>
#include <memory>
#include <vector>
#include "validation_lease.h"

#define WARMUPS         1000
#define NSPEEDS         9
#define LOCALSIZEX      128
#define LOCALSIZEY      1

/* dump output files for verification */
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int   nx;            /* no. of cells in x-direction */
  int   ny;            /* no. of cells in y-direction */
  int   maxIters;      /* no. of iterations */
  int   reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
 ** function prototypes
 */

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
    t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
    int** obstacles_ptr, float** av_vels_ptr);

/*
 ** The main calculation methods.
 ** timestep calls, in order, the functions:
 ** accelerate_flow(), propagate(), rebound() & collision()
 */
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(t_speed* cells_ptr, t_speed* tmp_cells_ptr,
    int* obstacles_ptr, float* av_vels_ptr);

/* Sum all the densities in the grid.
 ** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* input parameter file */
  char*    obstaclefile = NULL; /* input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstaclesHost = NULL;/* grid indicating which cells are blocked */
  float*   av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, 
      &tmp_cells, &obstaclesHost, &av_vels);

  // declare host arrays
  int Ny = params.ny;
  int Nx = params.nx;
  int MaxIters = params.maxIters;

  float *speedsHostS0 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS1 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS2 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS3 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS4 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS5 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS6 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS7 = (float*) malloc (sizeof(float)*Ny*Nx);
  float *speedsHostS8 = (float*) malloc (sizeof(float)*Ny*Nx);

  float *tot_up = (float*) malloc (sizeof(float) * (Ny/LOCALSIZEY) * (Nx/LOCALSIZEX) * MaxIters);
  int *tot_cellsp = (int*) malloc (sizeof(int) * (Ny/LOCALSIZEY) * (Nx/LOCALSIZEX) * MaxIters);

  // Init arrays
  /* loop over _all_ cells */
  for (int jj = 0; jj < Ny; jj++)
  {
    for (int ii = 0; ii < Nx; ii++)
    {
      speedsHostS0[ii + jj*Nx] = cells[ii + jj*Nx].speeds[0];
      speedsHostS1[ii + jj*Nx] = cells[ii + jj*Nx].speeds[1];
      speedsHostS2[ii + jj*Nx] = cells[ii + jj*Nx].speeds[2];
      speedsHostS3[ii + jj*Nx] = cells[ii + jj*Nx].speeds[3];
      speedsHostS4[ii + jj*Nx] = cells[ii + jj*Nx].speeds[4];
      speedsHostS5[ii + jj*Nx] = cells[ii + jj*Nx].speeds[5];
      speedsHostS6[ii + jj*Nx] = cells[ii + jj*Nx].speeds[6];
      speedsHostS7[ii + jj*Nx] = cells[ii + jj*Nx].speeds[7];
      speedsHostS8[ii + jj*Nx] = cells[ii + jj*Nx].speeds[8];
    }
  }

  // Original SoA double buffers, now owned by DMem.
  float* host_speed[] = {speedsHostS0, speedsHostS1, speedsHostS2, speedsHostS3, speedsHostS4, speedsHostS5, speedsHostS6, speedsHostS7, speedsHostS8};
  size_t n = size_t(Nx) * Ny;
  size_t partials = size_t(Nx/LOCALSIZEX) * (Ny/LOCALSIZEY) * MaxIters;
  {
    ValidationGPULease gpu_lease("D2Q9_GPU_LEASE");
    iris::Platform platform;
    if (platform.init(nullptr, nullptr, 1) != IRIS_SUCCESS) return 1;
    bool hip = false;
    for (int d=0; d<platform.ndevs(); ++d) {
      int backend=0, type=0; char name[256]={}; size_t bytes=0;
      iris_device_info(d, iris_backend, &backend, &bytes);
      iris_device_info(d, iris_type, &type, &bytes);
      iris_device_info(d, iris_name, name, &bytes);
      printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);
      hip |= backend==3 && type==256;
    }
    if (!hip) return 1;
    {
      std::vector<float> scratch[9];
      std::unique_ptr<iris::DMem> speed[9], tmp[9];
      for (int k=0;k<9;++k) {
        scratch[k].resize(n);
        speed[k].reset(new iris::DMem(host_speed[k],n*sizeof(float)));
        tmp[k].reset(new iris::DMem(scratch[k].data(),n*sizeof(float)));
      }
      iris::DMem obstacles(obstaclesHost,n*sizeof(int));
      iris::DMem sums(tot_up,partials*sizeof(float)), counts(tot_cellsp,partials*sizeof(int));
      float omega=params.omega, densityaccel=params.density*params.accel;
      size_t gws[]={size_t(Nx),size_t(Ny)}, lws[]={LOCALSIZEX,LOCALSIZEY};
      auto tic=std::chrono::steady_clock::now();
      for (int tt=0;tt<MaxIters;++tt) {
        if (tt==WARMUPS-1) tic=std::chrono::steady_clock::now();
        iris::Task task;
        void* args[26]; int info[26];
        for (int k=0;k<9;++k) {args[k]=speed[k].get();info[k]=iris_r;args[k+9]=tmp[k].get();info[k+9]=iris_w;}
        args[18]=&obstacles;info[18]=iris_r;
        args[19]=&sums;info[19]=iris_rw;args[20]=&counts;info[20]=iris_rw;
        args[21]=&densityaccel;info[21]=sizeof(densityaccel);
        args[22]=&omega;info[22]=sizeof(omega);
        args[23]=&Nx;info[23]=sizeof(Nx);args[24]=&Ny;info[24]=sizeof(Ny);args[25]=&tt;info[25]=sizeof(tt);
        if (task.kernel("d2q9_bgk",2,nullptr,gws,lws,26,args,info)!=IRIS_SUCCESS ||
            task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 1;
        for (int k=0;k<9;++k) speed[k].swap(tmp[k]);
      }
      auto time=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tic).count();
      printf("After warmup for %d iterations, ",WARMUPS);
      printf("average kernel execution time over %d iterations:\t\t\t%.6lf (us)\n",MaxIters-WARMUPS,time*1e-3/(MaxIters-WARMUPS));
      iris::Task flush;
      for (int k=0;k<9;++k) if(flush.flush_out(*speed[k])!=IRIS_SUCCESS) return 1;
      if(flush.flush_out(sums)!=IRIS_SUCCESS || flush.flush_out(counts)!=IRIS_SUCCESS || flush.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) return 1;
      if (MaxIters%2) for(int k=0;k<9;++k) memcpy(host_speed[k],scratch[k].data(),n*sizeof(float));
    }
    int errors=platform.error_count(); platform.finalize(); if(errors) return 1;
  } // Release validation GPU lease BEFORE CPU reduction and artifact output.

  float tot_u = 0;
  int tot_cells = 0;
  for (int tt = 0; tt < params.maxIters; tt++){
    tot_u = 0;
    tot_cells = 0;
    for(int i = 0; i < params.nx/LOCALSIZEX*params.ny/LOCALSIZEY; i++){
      tot_u += tot_up[i+tt*params.nx/LOCALSIZEX*params.ny/LOCALSIZEY];
      tot_cells += tot_cellsp[i+tt*params.nx/LOCALSIZEX*params.ny/LOCALSIZEY];
      //printf("%d %f %d\n", i, tot_u, tot_cells);
    }
    av_vels[tt] = tot_u/tot_cells;
  }

  // put answers back into cells
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      cells[ii + jj*params.nx].speeds[0] = speedsHostS0[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[1] = speedsHostS1[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[2] = speedsHostS2[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[3] = speedsHostS3[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[4] = speedsHostS4[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[5] = speedsHostS5[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[6] = speedsHostS6[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[7] = speedsHostS7[ii + jj*params.nx];
      cells[ii + jj*params.nx].speeds[8] = speedsHostS8[ii + jj*params.nx];
    }
  }

  if (const char* path=getenv("D2Q9_DUMP")) {
    FILE* f=fopen(path,"wb"); if(!f) return 1;
    if(fwrite(cells,sizeof(t_speed),size_t(Nx)*Ny,f)!=size_t(Nx)*Ny || fclose(f)) return 1;
  }
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstaclesHost));
  write_values(params, cells, obstaclesHost, av_vels);
  finalise(cells, tmp_cells, obstaclesHost, av_vels);

  free(speedsHostS0);
  free(speedsHostS1);
  free(speedsHostS2);
  free(speedsHostS3);
  free(speedsHostS4);
  free(speedsHostS5);
  free(speedsHostS6);
  free(speedsHostS7);
  free(speedsHostS8);
  free(tot_up);
  free(tot_cellsp);

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*params.nx].speeds[1]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[8]
            - (cells[ii + jj*params.nx].speeds[3]
              + cells[ii + jj*params.nx].speeds[6]
              + cells[ii + jj*params.nx].speeds[7]))
          / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*params.nx].speeds[2]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[6]
            - (cells[ii + jj*params.nx].speeds[4]
              + cells[ii + jj*params.nx].speeds[7]
              + cells[ii + jj*params.nx].speeds[8]))
          / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
    t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
    int** obstacles_ptr, float** av_vels_ptr){
  char   message[1024];  /* message buffer */
  FILE*  fp;             /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }
  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
   ** allocate memory.
   **
   ** remember C is pass-by-value, so we need to
   ** pass pointers into the initialise function.
   **
   ** nb we are allocating a 1D array, so that the
   ** memory will be contiguous.  We still want to
   ** index this memory as if it were a (row major
   ** ordered) 2D array, however.  We will perform
   ** some arithmetic using the row and column
   ** coordinates, inside the square brackets, when
   ** we want to access elements of this array.
   **
   ** note also that we are using a structure to
   ** hold an array of 'speeds'.  We will allocate
   ** a 1D array of these structs.
   */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = (int*) malloc (sizeof(int) * params->ny * params->nx);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
   ** allocate space to hold a record of the avarage velocities computed
   ** at each timestep
   */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(t_speed* cells_ptr, t_speed* tmp_cells_ptr,
    int* obstacles_ptr, float* av_vels_ptr)
{
  /*
   ** free up allocated memory
   */
  free(cells_ptr);
  free(tmp_cells_ptr);
  free(obstacles_ptr);
  free(av_vels_ptr);

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[8]
            - (cells[ii + jj*params.nx].speeds[3]
              + cells[ii + jj*params.nx].speeds[6]
              + cells[ii + jj*params.nx].speeds[7]))
          / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
            + cells[ii + jj*params.nx].speeds[5]
            + cells[ii + jj*params.nx].speeds[6]
            - (cells[ii + jj*params.nx].speeds[4]
              + cells[ii + jj*params.nx].speeds[7]
              + cells[ii + jj*params.nx].speeds[8]))
          / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
