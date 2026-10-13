/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include "iris_support.hpp"
#include <algorithm>
#include "file.h"
#include "config.hpp"

int main (int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <input filename> <output filename>\n", argv[0]);
    return 1;
  }
  
  char* inputFileName = argv[1];
  char* outputFileName = argv[2];

  int numX, numK;		/* Number of X and K values */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;

  /* Read in data */
  inputData(inputFileName,
	    &numK, &numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

  printf("%d pixels in output; %d samples in trajectory\n", numX, numK);

  /* Create CPU data structures */
  phiMag=(float*)memalign(16,numK*sizeof(float));
  Qr=(float*)calloc(numX,sizeof(float)); Qi=(float*)calloc(numX,sizeof(float));
  iris::Platform platform;
  int device=init_iris(platform,argc,argv);

  /* GPU section 1 (precompute PhiMag) */
  /* Mirror several data structures on the device */
  iris::DMem phiR_d(phiR,numK*sizeof(float)), phiI_d(phiI,numK*sizeof(float));
  iris::DMem phiMag_d(phiMag,numK*sizeof(float));
  auto start = std::chrono::steady_clock::now();

  {
    void* p[]={ &phiR_d,&phiI_d,&phiMag_d,&numK };
    int info[]={ iris_r,iris_r,iris_w,sizeof(numK) };
    launch_iris(device,"ComputePhiMag_GPU",((numK+255)/256)*256,256,4,p,info,&phiMag_d);
  }


  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computePhiMag time: %f s\n", time * 1e-9);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  /* GPU section 2 */
  iris::DMem x_d(x,numX*sizeof(float)),y_d(y,numX*sizeof(float)),z_d(z,numX*sizeof(float));
  iris::DMem Qr_d(Qr,numX*sizeof(float)),Qi_d(Qi,numX*sizeof(float));
  iris::DMem values(kVals,numK*sizeof(kValues));
  start = std::chrono::steady_clock::now();

  for(int base=0;base<numK;base+=KERNEL_Q_K_ELEMS_PER_GRID) {
    void* p[]={ &numK,&base,&x_d,&y_d,&z_d,&Qr_d,&Qi_d,&values,&numX };
    int info[]={ sizeof(numK),sizeof(base),iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_r,sizeof(numX) };
    launch_iris(device,"ComputeQ_GPU",((numX+255)/256)*256,256,9,p,info);
  }
  flush_iris(device,Qr_d); flush_iris(device,Qi_d);


  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("computeQ time: %f s\n", time * 1e-9);

  outputData(outputFileName, Qr, Qi, numX);

  free(phiMag);
  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);
  const int errors=platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}
