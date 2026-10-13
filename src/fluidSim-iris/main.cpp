#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include "dmem_runtime.hpp"
#include "utils.h"
#include "dispatch.hpp"
#include "reference.h"

int main(int argc, char * argv[])
{
  if (argc != 2) {
    printf("Usage %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]); // Simulation iterations
  const int lbm_width = 256;            // Dimension of LBM simulation area
  const int lbm_height = 256;

  int dims[2] = {lbm_width, lbm_height};
  size_t temp = dims[0] * dims[1];

   // Directions
   double e[9][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {-1,-1}, {1,-1}};
   
   // Weights
   double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

  // Omega
  const double omega = 1.2f;

  double8 dirX, dirY; // Directions

  // host inputs
  double *h_if0 = (double*)malloc(sizeof(double) * temp);
  double *h_if1234 = (double*)malloc(sizeof(double4) * temp);
  double *h_if5678 = (double*)malloc(sizeof(double4) * temp);

#ifdef VERIFY
  // Reference outputs
  double *v_of0 = (double*)malloc(sizeof(double) * temp);
  double *v_of1234 = (double*)malloc(sizeof(double4) * temp);
  double *v_of5678 = (double*)malloc(sizeof(double4) * temp);
#endif

  // Host outputs
  double *h_of0 = (double*)malloc(sizeof(double) * temp);
  double *h_of1234 = (double*)malloc(sizeof(double4) * temp);
  double *h_of5678 = (double*)malloc(sizeof(double4) * temp);

  // Cell Type - Boundary = 1 or Fluid = 0
  bool *h_type = (bool*)malloc(sizeof(bool) * temp);
  
  // Density
  double *rho = (double*)malloc(sizeof(double) * temp);
  
  // Velocity
  double2 *u = (double2*)malloc(sizeof(double2) * temp);

  // Initial velocity is nonzero for verifying host and device results
  double u0[2] = {0.01, 0.01};

  srand(123);
  for (int y = 0; y < dims[1]; y++)
  {
    for (int x = 0; x < dims[0]; x++)
    {
      int pos = x + y * dims[0];

      // Random values for verification 
      double den = rand() % 10 + 1; 

      // Initialize the velocity buffer
      u[pos].x = u0[0];
      u[pos].y = u0[1];

      h_if0[pos]            = computefEq(den, w[0], e[0], u0);
      h_if1234[pos * 4 + 0] = computefEq(den, w[1], e[1], u0);
      h_if1234[pos * 4 + 1] = computefEq(den, w[2], e[2], u0);
      h_if1234[pos * 4 + 2] = computefEq(den, w[3], e[3], u0);
      h_if1234[pos * 4 + 3] = computefEq(den, w[4], e[4], u0);
      h_if5678[pos * 4 + 0] = computefEq(den, w[5], e[5], u0);
      h_if5678[pos * 4 + 1] = computefEq(den, w[6], e[6], u0);
      h_if5678[pos * 4 + 2] = computefEq(den, w[7], e[7], u0);
      h_if5678[pos * 4 + 3] = computefEq(den, w[8], e[8], u0);

      // Initialize boundary cells
      if (x == 0 || x == (dims[0] - 1) || y == 0 || y == (dims[1] - 1))
        h_type[pos] = 1;

      // Initialize fluid cells
      else
        h_type[pos] = 0;
    }
  }

  // initialize direction vectors
  dirX.s0 = e[1][0]; dirY.s0 = e[1][1];
  dirX.s1 = e[2][0]; dirY.s1 = e[2][1];
  dirX.s2 = e[3][0]; dirY.s2 = e[3][1];
  dirX.s3 = e[4][0]; dirY.s3 = e[4][1];
  dirX.s4 = e[5][0]; dirY.s4 = e[5][1];
  dirX.s5 = e[6][0]; dirY.s5 = e[6][1];
  dirX.s6 = e[7][0]; dirY.s6 = e[7][1];
  dirX.s7 = e[8][0]; dirY.s7 = e[8][1];

#ifdef VERIFY
  reference (iterations, omega, dims, h_type, rho, e, w,
             h_if0, h_if1234, h_if5678,
             v_of0, v_of1234, v_of5678);
#endif

  fluidSim (iterations, omega, dims, h_type, u, rho, dirX, dirY, w,
            h_if0, h_if1234, h_if5678,
            h_of0, h_of1234, h_of5678);

#ifdef VERIFY
  verify(dims, h_of0, h_of1234, h_of5678,
               v_of0, v_of1234, v_of5678);

  if(v_of0)    free(v_of0);
  if(v_of1234) free(v_of1234);
  if(v_of5678) free(v_of5678);
#endif
  
  if (const char *path = getenv("HECBENCH_DUMP_OUTPUT")) {
    FILE *out = fopen(path, "wb");
    if (!out) return 2;
    fwrite(h_of0, sizeof(double), temp, out);
    fwrite(h_of1234, sizeof(double), temp * 4, out);
    fwrite(h_of5678, sizeof(double), temp * 4, out);
    fclose(out);
  }

  if(h_if0)    free(h_if0);
  if(h_if1234) free(h_if1234);
  if(h_if5678) free(h_if5678);
  if(h_of0)    free(h_of0);
  if(h_of1234) free(h_of1234);
  if(h_of5678) free(h_of5678);
  if(h_type)   free(h_type);
  if(rho)      free(rho);
  if(u)        free(u);

  iris_finish();
  return 0;
}

#define GROUP_SIZE 256

void fluidSim (
  const int iterations,
  const double omega,
  const int *dims,
  const bool *h_type,
  double2 *u,
  double *rho,
  const double8 dirX,
  const double8 dirY,
  const double w[9],
        double *h_if0,
        double *h_if1234,
        double *h_if5678,
        double *h_of0,
        double *h_of1234,
        double *h_of5678)
{
  int groupSize = GROUP_SIZE;
  size_t dbl_size = dims[0] * dims[1] * sizeof(double);
  size_t dbl4_size = dims[0] * dims[1] * sizeof(double4);
  size_t bool_size = dims[0] * dims[1] * sizeof(bool);

  // allocate and initialize device buffers
  double *d_if0, *d_of0;
  double4 *d_if1234, *d_if5678;
  double4 *d_of1234, *d_of5678;

  iris_alloc((void**)&d_if0, dbl_size); 
  iris_copy(d_if0, h_if0, dbl_size, iris_h2d);

  iris_alloc((void**)&d_of0, dbl_size); 
  iris_alloc((void**)&d_if1234, dbl4_size);
  iris_alloc((void**)&d_if5678, dbl4_size);
  iris_alloc((void**)&d_of1234, dbl4_size);
  iris_alloc((void**)&d_of5678, dbl4_size);

  iris_copy(d_if1234, (double4*)h_if1234, dbl4_size, iris_h2d);
  iris_copy(d_if5678, (double4*)h_if5678, dbl4_size, iris_h2d);

  iris_copy(d_of0, d_if0, dbl_size, iris_d2d);
  iris_copy(d_of1234, d_if1234, dbl4_size, iris_d2d);
  iris_copy(d_of5678, d_if5678, dbl4_size, iris_d2d);

  bool *d_type;
  iris_alloc((void**)&d_type, bool_size);
  iris_copy(d_type, h_type, bool_size, iris_h2d);

  double *d_weight;
  iris_alloc((void**)&d_weight, sizeof(double)*9);
  iris_copy(d_weight, w, sizeof(double)*9, iris_h2d);

  dim3 grids (groupSize, 1);
  dim3 blocks (dims[0]/groupSize, dims[1]);

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  for(int i = 0; i < iterations; ++i) {
    dispatch_lbm( dim3(grids), dim3(blocks), 
        d_if0, d_of0, d_if1234, d_of1234,
        d_if5678, d_of5678, d_type,
        dirX, dirY, d_weight, omega
    );

    // Swap device buffers
    double *temp0 = d_of0;
    double4 *temp1234 = d_of1234;
    double4 *temp5678 = d_of5678;

    d_of0 = d_if0;
    d_of1234 = d_if1234;
    d_of5678 = d_if5678;

    d_if0 = temp0;
    d_if1234 = temp1234;
    d_if5678 = temp5678;
  }

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iterations);

  iris_copy(h_of0, d_if0, dbl_size, iris_d2h);
  iris_copy((double4*)h_of1234, d_if1234, dbl4_size, iris_d2h);
  iris_copy((double4*)h_of5678, d_if5678, dbl4_size, iris_d2h);

  iris_free(d_if0);
  iris_free(d_of0);
  iris_free(d_if1234);
  iris_free(d_of1234);
  iris_free(d_if5678);
  iris_free(d_of5678);
  iris_free(d_type);
  iris_free(d_weight);
}
