/** GPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_gpu.cu
 *
 * \author Kyle E. Niemeyer
 * \date 09/27/2012
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity on a GPU (via CUDA), using the finite difference method.
 * To change the grid resolution, modify "NUM". In addition, the problem is controlled
 * by the Reynolds number ("Re_num").
 * 
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, Philadelphia,
 * PA, 1998.
 * 
 * Boundary conditions:
 * u = 0 and v = 0 at x = 0, x = L, y = 0
 * u = ustar at y = H
 * v = 0 at y = H
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iris/iris.hpp>
struct dim3{size_t x,y;dim3(size_t a,size_t b):x(a),y(b){}};

/** Problem size along one side; total number of cells is this squared */
#ifndef NUM
#define NUM 512
#endif

// block size
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

/** Double precision */
#define DOUBLE

#ifdef DOUBLE
#define Real double

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0

#define SMALL 1.0e-10;

/** Reynolds number */
const Real Re_num = 1000.0;

/** SOR relaxation parameter */
const Real omega = 1.7;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9;

/** Safety factor for time step modification */
const Real tau = 0.5;

/** Body forces in x- and y- directions */
const Real gx = 0.0;
const Real gy = 0.0;

/** Domain size (non-dimensional) */
#define xLength 1.0
#define yLength 1.0
#else
#define Real float

// replace double functions with float versions
#undef fmin
#define fmin fminf
#undef fmax
#define fmax fmaxf
#undef fabs
#define fabs fabsf
#undef sqrt
#define sqrt sqrtf

#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SMALL 1.0e-10f;

/** Reynolds number */
const Real Re_num = 1000.0f;

/** SOR relaxation parameter */
const Real omega = 1.7f;

/** Discretization mixture parameter (gamma) */
const Real mix_param = 0.9f;

/** Safety factor for time step modification */
const Real tau = 0.5f;

/** Body forces in x- and y- directions */
const Real gx = 0.0f;
const Real gy = 0.0f;

/** Domain size (non-dimensional) */
#define xLength 1.0f
#define yLength 1.0f
#endif

/** Mesh sizes */
const Real dx = xLength / NUM;
const Real dy = yLength / NUM;

/** Max macro (type safe, from GNU) */
//#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

/** Min macro (type safe) */
//#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

// map two-dimensional indices to one-dimensional memory
#define u(I, J) u[((I) * ((NUM) + 2)) + (J)]
#define v(I, J) v[((I) * ((NUM) + 2)) + (J)]
#define F(I, J) F[((I) * ((NUM) + 2)) + (J)]
#define G(I, J) G[((I) * ((NUM) + 2)) + (J)]
#define pres_red(I, J) pres_red[((I) * ((NUM_2) + 2)) + (J)]
#define pres_black(I, J) pres_black[((I) * ((NUM_2) + 2)) + (J)]

///////////////////////////////////////////////////////////////////////////////

void set_BCs_host (Real* u, Real* v) 
{
  int ind;

  // loop through rows and columns
  for (ind = 0; ind < NUM + 2; ++ind) {

    // left boundary
    u(0, ind) = ZERO;
    v(0, ind) = -v(1, ind);

    // right boundary
    u(NUM, ind) = ZERO;
    v(NUM + 1, ind) = -v(NUM, ind);

    // bottom boundary
    u(ind, 0) = -u(ind, 1);
    v(ind, 0) = ZERO;

    // top boundary
    u(ind, NUM + 1) = TWO - u(ind, NUM);
    v(ind, NUM) = ZERO;

    if (ind == NUM) {
      // left boundary
      u(0, 0) = ZERO;
      v(0, 0) = -v(1, 0);
      u(0, NUM + 1) = ZERO;
      v(0, NUM + 1) = -v(1, NUM + 1);

      // right boundary
      u(NUM, 0) = ZERO;
      v(NUM + 1, 0) = -v(NUM, 0);
      u(NUM, NUM + 1) = ZERO;
      v(NUM + 1, NUM + 1) = -v(NUM, NUM + 1);

      // bottom boundary
      u(0, 0) = -u(0, 1);
      v(0, 0) = ZERO;
      u(NUM + 1, 0) = -u(NUM + 1, 1);
      v(NUM + 1, 0) = ZERO;

      // top boundary
      u(0, NUM + 1) = TWO - u(0, NUM);
      v(0, NUM) = ZERO;
      u(NUM + 1, NUM + 1) = TWO - u(NUM + 1, NUM);
      v(ind, NUM + 1) = ZERO;
    } // end if

  } // end for

} // end set_BCs_host

///////////////////////////////////////////////////////////////////////////////
int main (int argc, char *argv[])
{
  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS)return 1;
  bool gpu=false;
  for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }
  if(!gpu)return 1;
  // iterations for Red-Black Gauss-Seidel with SOR
  int iter = 0;
  const int it_max = 1000000;

  // SOR iteration tolerance
  const Real tol = 0.001;

  // time range
  const Real time_start = 0.0;
  const Real time_end = 0.001; //20.0;

  // initial time step size
  Real dt = 0.02;

  int size = (NUM + 2) * (NUM + 2);
  int size_pres = ((NUM / 2) + 2) * (NUM + 2);

  // arrays for pressure and velocity
  Real* F;
  Real* u;
  Real* G;
  Real* v;

  F = (Real *) calloc (size, sizeof(Real));
  u = (Real *) calloc (size, sizeof(Real));
  G = (Real *) calloc (size, sizeof(Real));
  v = (Real *) calloc (size, sizeof(Real));

  for (int i = 0; i < size; ++i) {
    F[i] = ZERO;
    u[i] = ZERO;
    G[i] = ZERO;
    v[i] = ZERO;
  }

  // arrays for pressure
  Real* pres_red;
  Real* pres_black;

  pres_red = (Real *) calloc (size_pres, sizeof(Real));
  pres_black = (Real *) calloc (size_pres, sizeof(Real));

  for (int i = 0; i < size_pres; ++i) {
    pres_red[i] = ZERO;
    pres_black[i] = ZERO;
  }

  // print problem size
  printf("Problem size: %d x %d \n", NUM, NUM);

  // block and grid dimensions

  // boundary conditions kernel
  dim3 block_bcs (BLOCK_SIZE, 1);
  dim3 grid_bcs (NUM / BLOCK_SIZE, 1);

  // pressure kernel
  dim3 block_pr (BLOCK_SIZE, 1);
  dim3 grid_pr (NUM / (2 * BLOCK_SIZE), NUM);

  // block and grid dimensions for F
  dim3 block_F (BLOCK_SIZE, 1);
  dim3 grid_F (NUM / BLOCK_SIZE, NUM);

  // block and grid dimensions for G
  dim3 block_G (BLOCK_SIZE, 1);
  dim3 grid_G (NUM / BLOCK_SIZE, NUM);

  // horizontal pressure boundary conditions
  dim3 block_hpbc (BLOCK_SIZE, 1);
  dim3 grid_hpbc (NUM / (2 * BLOCK_SIZE), 1);

  // vertical pressure boundary conditions
  dim3 block_vpbc (BLOCK_SIZE, 1);
  dim3 grid_vpbc (NUM / (2 * BLOCK_SIZE), 1);

  // residual variable
  Real* res;

  int size_res = grid_pr.x * grid_pr.y;
  res = (Real *) calloc (size_res, sizeof(Real));

  // variables to store maximum velocities
  Real* max_u_arr;
  Real* max_v_arr;
  int size_max = grid_pr.x * grid_pr.y;

  max_u_arr = (Real *) calloc (size_max, sizeof(Real));
  max_v_arr = (Real *) calloc (size_max, sizeof(Real));

  // pressure sum
  Real* pres_sum;
  pres_sum = (Real *) calloc (size_res, sizeof(Real));

  // set initial BCs
  set_BCs_host (u, v);

  Real max_u = SMALL;
  Real max_v = SMALL;
  // get max velocity for initial values (including BCs)
  #pragma unroll
  for (int col = 0; col < NUM + 2; ++col) {
    #pragma unroll
    for (int row = 1; row < NUM + 2; ++row) {
      max_u = fmax(max_u, fabs( u(col, row) ));
    }
  }

  #pragma unroll
  for (int col = 1; col < NUM + 2; ++col) {
    #pragma unroll
    for (int row = 0; row < NUM + 2; ++row) {
      max_v = fmax(max_v, fabs( v(col, row) ));
    }
  }

  iris::DMem u_d(u,size*sizeof(Real));
  iris::DMem F_d(F,size*sizeof(Real));
  iris::DMem v_d(v,size*sizeof(Real));
  iris::DMem G_d(G,size*sizeof(Real));
  iris::DMem pres_red_d(pres_red,size_pres*sizeof(Real));
  iris::DMem pres_black_d(pres_black,size_pres*sizeof(Real));
  iris::DMem pres_sum_d(pres_sum,size_res*sizeof(Real));
  iris::DMem res_d(res,size_res*sizeof(Real));
  iris::DMem max_u_d(max_u_arr,size_max*sizeof(Real));
  iris::DMem max_v_d(max_v_arr,size_max*sizeof(Real));

  Real time = time_start;

  // time-step size based on grid and Reynolds number
  Real dt_Re = 0.5 * Re_num / ((1.0 / (dx * dx)) + (1.0 / (dy * dy)));

  auto start = std::chrono::steady_clock::now();

  // time iteration loop
  while (time < time_end) {

    // calculate time step based on stability and CFL
    dt = fmin((dx / max_u), (dy / max_v));
    dt = tau * fmin(dt_Re, dt);

    if ((time + dt) >= time_end) {
      dt = time_end - time;
    }

    // calculate F and G    
    {
      iris::Task task;size_t gws[]={grid_F.x*block_F.x,grid_F.y*block_F.y},lws[]={block_F.x,block_F.y};
      void* args[]={&dt,&u_d,&v_d,&F_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw};
      if(task.kernel("calculate_F",2,nullptr,gws,lws,4,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }
    {
      iris::Task task;size_t gws[]={grid_G.x*block_G.x,grid_G.y*block_G.y},lws[]={block_G.x,block_G.y};
      void* args[]={&dt,&u_d,&v_d,&G_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw};
      if(task.kernel("calculate_G",2,nullptr,gws,lws,4,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

    // get L2 norm of initial pressure
    {
      iris::Task task;size_t gws[]={grid_pr.x*block_pr.x,grid_pr.y*block_pr.y},lws[]={block_pr.x,block_pr.y};
      void* args[]={&pres_red_d,&pres_black_d,&pres_sum_d};int info[]={iris_rw,iris_rw,iris_rw};
      if(task.kernel("sum_pressure",2,nullptr,gws,lws,3,args,info)!=IRIS_SUCCESS)return 2;
      task.flush_out(pres_sum_d);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

    Real p0_norm = ZERO;
    #pragma unroll
    for (int i = 0; i < size_res; ++i) {
      p0_norm += pres_sum[i];
    }

    p0_norm = sqrt(p0_norm / ((Real)(NUM * NUM)));
    if (p0_norm < 0.0001) {
      p0_norm = 1.0;
    }

    // ensure all kernels are finished
    //cudaDeviceSynchronize();

    Real norm_L2;

    // calculate new pressure
    // red-black Gauss-Seidel with SOR iteration loop
    for (iter = 1; iter <= it_max; ++iter) {

      // set pressure boundary conditions
      {
      iris::Task task;size_t gws[]={grid_hpbc.x*block_hpbc.x,grid_hpbc.y*block_hpbc.y},lws[]={block_hpbc.x,block_hpbc.y};
      void* args[]={&pres_red_d,&pres_black_d};int info[]={iris_rw,iris_rw};
      if(task.kernel("set_horz_pres_BCs",2,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }
      {
      iris::Task task;size_t gws[]={grid_vpbc.x*block_vpbc.x,grid_vpbc.y*block_vpbc.y},lws[]={block_vpbc.x,block_vpbc.y};
      void* args[]={&pres_red_d,&pres_black_d};int info[]={iris_rw,iris_rw};
      if(task.kernel("set_vert_pres_BCs",2,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

      // ensure kernel finished
      //cudaDeviceSynchronize();

      // update red cells
      {
      iris::Task task;size_t gws[]={grid_pr.x*block_pr.x,grid_pr.y*block_pr.y},lws[]={block_pr.x,block_pr.y};
      void* args[]={&dt,&F_d,&G_d,&pres_black_d,&pres_red_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw,iris_rw};
      if(task.kernel("red_kernel",2,nullptr,gws,lws,5,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

      // ensure red kernel finished
      //cudaDeviceSynchronize();

      // update black cells
      {
      iris::Task task;size_t gws[]={grid_pr.x*block_pr.x,grid_pr.y*block_pr.y},lws[]={block_pr.x,block_pr.y};
      void* args[]={&dt,&F_d,&G_d,&pres_red_d,&pres_black_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw,iris_rw};
      if(task.kernel("black_kernel",2,nullptr,gws,lws,5,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

      // ensure red kernel finished
      //cudaDeviceSynchronize();

      // calculate residual values
      {
      iris::Task task;size_t gws[]={grid_pr.x*block_pr.x,grid_pr.y*block_pr.y},lws[]={block_pr.x,block_pr.y};
      void* args[]={&dt,&F_d,&G_d,&pres_red_d,&pres_black_d,&res_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw};
      if(task.kernel("calc_residual",2,nullptr,gws,lws,6,args,info)!=IRIS_SUCCESS)return 2;
      task.flush_out(res_d);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

      // transfer residual value(s) back to CPU

      norm_L2 = ZERO;
      #pragma unroll
      for (int i = 0; i < size_res; ++i) {
        norm_L2 += res[i];
      }

      // calculate residual
      norm_L2 = sqrt(norm_L2 / ((Real)(NUM * NUM))) / p0_norm;

      // if tolerance has been reached, end SOR iterations
      if (norm_L2 < tol) {
        break;
      }  
    } // end for

    printf("Time = %f, delt = %e, iter = %i, res = %e\n", time + dt, dt, iter, norm_L2);

    // calculate new velocities and transfer maximums back

    {
      iris::Task task;size_t gws[]={grid_pr.x*block_pr.x,grid_pr.y*block_pr.y},lws[]={block_pr.x,block_pr.y};
      void* args[]={&dt,&F_d,&pres_red_d,&pres_black_d,&u_d,&max_u_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw};
      if(task.kernel("calculate_u",2,nullptr,gws,lws,6,args,info)!=IRIS_SUCCESS)return 2;
      task.flush_out(max_u_d);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

    {
      iris::Task task;size_t gws[]={grid_pr.x*block_pr.x,grid_pr.y*block_pr.y},lws[]={block_pr.x,block_pr.y};
      void* args[]={&dt,&G_d,&pres_red_d,&pres_black_d,&v_d,&max_v_d};int info[]={sizeof(dt),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw};
      if(task.kernel("calculate_v",2,nullptr,gws,lws,6,args,info)!=IRIS_SUCCESS)return 2;
      task.flush_out(max_v_d);
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

    // get maximum u- and v- velocities
    max_v = SMALL;
    max_u = SMALL;

    #pragma unroll
    for (int i = 0; i < size_max; ++i) {
      Real test_u = max_u_arr[i];
      max_u = fmax(max_u, test_u);

      Real test_v = max_v_arr[i];
      max_v = fmax(max_v, test_v);
    }

    // set velocity boundary conditions
    {
      iris::Task task;size_t gws[]={grid_bcs.x*block_bcs.x,grid_bcs.y*block_bcs.y},lws[]={block_bcs.x,block_bcs.y};
      void* args[]={&u_d,&v_d};int info[]={iris_rw,iris_rw};
      if(task.kernel("set_BCs",2,nullptr,gws,lws,2,args,info)!=IRIS_SUCCESS)return 2;
      
      if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
    }

    // increase time
    time += dt;

    // single time step
    //break;

  } // end while

  auto end = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nTotal execution time of the iteration loop: %f (s)\n", elapsed_time * 1e-9f);

  // transfer final temperature values back




  {
    iris::Task fetch;fetch.flush_out(u_d);fetch.flush_out(v_d);fetch.flush_out(pres_red_d);fetch.flush_out(pres_black_d);
    if(fetch.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
  }
  if(const char* path=getenv("HECBENCH_ARTIFACT")) {
    FILE* out=fopen(path,"wb");if(!out)return 3;
    fwrite(u,sizeof(Real),size,out);fwrite(v,sizeof(Real),size,out);
    fwrite(pres_red,sizeof(Real),size_pres,out);fwrite(pres_black,sizeof(Real),size_pres,out);fclose(out);
  }
  // write data to file
  FILE * pfile;
  pfile = fopen("velocity_gpu.dat", "w");
  fprintf(pfile, "#x\ty\tu\tv\n");
  if (pfile != NULL) {
    for (int row = 0; row < NUM; ++row) {
      for (int col = 0; col < NUM; ++col) {

        Real u_ij = u[(col * NUM) + row];
        Real u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        } else {
          u_im1j = u[(col - 1) * NUM + row];
        }

        u_ij = (u_ij + u_im1j) / 2.0;

        Real v_ij = v[(col * NUM) + row];
        Real v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        } else {
          v_ijm1 = v[(col * NUM) + row - 1];
        }

        v_ij = (v_ij + v_ijm1) / 2.0;

        fprintf(pfile, "%f\t%f\t%f\t%f\n", ((Real)col + 0.5) * dx, ((Real)row + 0.5) * dy, u_ij, v_ij);
      }
    }
  }

  fclose(pfile);

  // free device memory











  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  free(max_u_arr);
  free(max_v_arr);
  free(res);
  free(pres_sum);

  int errors=platform.error_count();platform.finalize();return errors ? 1 : 0;
}
