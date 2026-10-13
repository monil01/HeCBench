//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////

#include "utils.h"
#include "launch.h"
#include "prng_host.h"

int main(int argc, char **argv){
  iris::Platform platform; platform.init(&argc, &argv, 1);
  prove_backend();

  int L         = 32;
  int R         = 1;
  int atrials   = 1;
  int ains      = 1;
  int apts      = 1;
  int ams       = 1;
  uint64_t seed = 2;  
  float TR      = 0.1f; 
  float dT      = 0.1f; 
  float h       = 0.1f;

  for (int i=0; i<argc; i++) {
    /* lattice size and number of replicas */
    if(strcmp(argv[i],"-l") == 0){
      L = atoi(argv[i+1]);  
      if ( (L % 32) != 0 ) {
        fprintf(stderr, "lattice dimensional size must be multiples of 32");
        exit(1);
      }
      R = atoi(argv[i+2]);
    }
    /* get TR and dT */
    else if(strcmp(argv[i],"-t") == 0){
      TR = atof(argv[i+1]);
      dT = atof(argv[i+2]);
    }
    /* the magnetic field constant */
    else if(strcmp(argv[i],"-h") == 0){
      h = atof(argv[i+1]);
    }
    /* adaptative dt parameters (number of trials, insertions, tempering, simulation  */
    else if(strcmp(argv[i], "-a") == 0){
      atrials = atoi(argv[i+1]);
      ains = atoi(argv[i+2]);
      apts = atoi(argv[i+3]);
      ams = atoi(argv[i+4]);
    }
    /* seed for random number generation */
    else if(strcmp(argv[i],"-z") == 0){
      seed = atol(argv[i+1]);
    }
  }

  /* total number of spins per replica */
  int N = (L)*(L)*(L);

  /* compute Ra to be the final size Ra = R + TL */
  int Ra = R + (atrials * ains);

  /* active replicas per gpu */
  int ar = R;

  /* replica pool per gpu */
  int rpool = Ra;


  /* parameter seed */
  uint64_t hpcgs, hpcgi;

  gpu_pcg32_srandom_r(&hpcgs, &hpcgi, seed, 1);
  seed = gpu_pcg32_random_r(&hpcgs, &hpcgi);

  /* build the space of computation for the lattices */

  /* build the space of computation for random numbers and lattice simulation */

  /* T is a sorted temp array */
  float* T = (float*)malloc(sizeof(float) * Ra);

  /* allocate the replica pool */
  iris::DMem** mdlat = new iris::DMem*[rpool];
  /* per temperature counter array */
  float* aex = (float*) malloc(sizeof(float) * rpool);
  /* per temperature counter array */
  float* aavex = (float*)malloc(sizeof(float) * rpool);
  /* exchange energies */
  float* aexE = (float*)malloc(sizeof(float) * rpool);

  /* PRNG states volume, one state per thread */
  iris::DMem** apcga = new iris::DMem*[rpool];
  iris::DMem** apcgb = new iris::DMem*[rpool];

  /* fragmented indices for replicas temperature sorted */
  findex_t* arts = (findex_t*)malloc(sizeof(findex_t) * rpool);
  /* fragmented indices for temperatures replica sorted */
  findex_t* atrs = (findex_t*)malloc(sizeof(findex_t) * rpool);
  /* fragmented temperatures sorted */
  float* aT = (float*)malloc(sizeof(float) * rpool);

  /* malloc device magnetic field */
  iris::DMem* dH;
  dH = create_mem(sizeof(int)*N);

  /* malloc device energy reductions */
  iris::DMem* dE;
  dE = create_mem(sizeof(float)*rpool);

  /* malloc the data for 'r' replicas on each GPU */
  for (int k = 0; k < rpool; ++k) {
    mdlat[k] = create_mem(sizeof(int) * N);
    apcga[k] = create_mem((N/4) * sizeof(uint64_t));
    apcgb[k] = create_mem((N/4) * sizeof(uint64_t));
    // offset and sequence approach
    launch_setup(apcga[k], apcgb[k], N/4, 
        seed + N/4 * k, k);
  }

  /* host memory setup for each replica */
  for(int i = 0; i < R; i++){
    /* array of temperatures increasing order */
    T[i] = TR - (R-1 - i)*dT;
  }

  int count = 0;
  for(int j = 0; j < ar; ++j){
    arts[j] = atrs[j] = (findex_t){0, j};
    aT[j] = TR - (float)(R - 1 - count) * dT;
    aex[j] = 0;
    ++count;
  }

  /* print parameters */
  printf("\tparameters:{\n");
  printf("\t\tL:                            %i\n", L);
  printf("\t\tvolume:                       %i\n", N);
  printf("\t\t[TR,dT]:                      [%f, %f]\n", TR, dT);
  printf("\t\t[atrials, ains, apts, ams]:   [%i, %i, %i, %i]\n", atrials, ains, apts, ams);
  printf("\t\tmag_field h:                  %f\n", h);
  printf("\t\treplicas:                     %i\n", R);
  printf("\t\tseed:                         %lu\n", seed);

  /* find good temperature distribution */
  FILE *fw = fopen("trials.dat", "w");
  fprintf(fw, "trial  av  min max\n");

  long total_ktime = 0;

  auto start = std::chrono::steady_clock::now();

  /* each adaptation iteration improves the temperature distribution */
  for (int trial = 0; trial < atrials; ++trial) {

    /* progress printing */
    printf("[trial %i of %i]\n", trial+1, atrials); fflush(stdout);

    /* distribution for H */
    launch_random(dH, N, apcga[0], apcgb[0]);  
    
    /* reset ex counters */
    reset_array(aex, rpool, 0.0f);

    /* reset average ex counters */
    reset_array(aavex, rpool, 0.0f);

    /* reset gpu data with a new seed from the sequential PRNG */
    seed = gpu_pcg32_random_r(&hpcgs, &hpcgi);

    for (int k = 0; k < ar; ++k) {
      launch_reset_i(mdlat[k], N, 1);
      

      launch_setup(apcga[k], apcgb[k], N/4, 
          seed + (uint64_t)(N/4 * k), k);
      
    }

    /* parallel tempering */
    for(int p = 0; p < apts; ++p) {

      auto kstart = std::chrono::steady_clock::now();

      /* metropolis simulation */
      for(int i = 0; i < ams; ++i) {
        for(int k = 0; k < ar; ++k) {

          launch_metro(N, L, mdlat[k], dH, h, 
              -2.0f/aT[atrs[k].i], apcga[k], apcgb[k], 0);
        }

        
        

        for(int k = 0; k < ar; ++k) {
          launch_metro(N, L, mdlat[k], dH, h, 
              -2.0f/aT[atrs[k].i], apcga[k], apcgb[k], 1);
        }

        
        
      }

      auto kend = std::chrono::steady_clock::now();
      total_ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

      /* compute energies for exchange */
      // adapt_ptenergies(s, tid);
      launch_reset_f (dE, ar, 0.0f);
      

      /* compute one energy reduction for each replica */
        
      for(int k = 0; k < ar; ++k){
        /* launch reduction kernel for k-th replica */
        launch_energy(mdlat[k], L, dE, dH, h, k);
        
        
      }
      dE->fetch(aexE, ar*sizeof(float));

      /* exchange phase */
      double delta = 0.0;
      findex_t fnow, fleft;
      fnow.f = 0;  // the f field is always 0 for a single GPU
      fnow.i = ar-1;
      /* traverse in reverse temperature order */
      for (int k = R-1; k > 0; --k) {
        if((k % 2) == (p % 2)){
          fgoleft(&fnow, ar);
          continue;
        }
        fleft = fgetleft(fnow, ar);

        delta = (1.0f/aT[fnow.i] - 1.0f/aT[fleft.i]) *
          (aexE[arts[fleft.i].i] - aexE[arts[fnow.i].i]);

        double randme = gpu_rand01(&hpcgs, &hpcgi);

        if( delta < 0.0 || randme < exp(-delta) ){
          //adapt_swap(s, fnow, fleft);
          findex_t t1 = arts[fnow.i];
          findex_t t2 = arts[fleft.i];
          findex_t taux = atrs[t1.i];
          findex_t raux = arts[fnow.i];

          /* swap rts */
          arts[fnow.i] = arts[fleft.i];
          arts[fleft.i] = raux;

          /* swap trs */
          atrs[t1.i] = atrs[t2.i];
          atrs[t2.i] = taux;

          /* this array is temp sorted */
          aex[fnow.i] += 1.0f;
        }
        fgoleft(&fnow, ar);
      }
      printf("\rpt........%i%%", 100 * (p + 1)/apts); fflush(stdout);
    }

    double avex = 0;
    for(int k = 1; k < ar; ++k){
      avex += aavex[k] = 2.0 * aex[k] / (double)apts;
    }
    avex /= (double)(R-1);

    double minex = 1;
    for(int k = 1; k < ar; ++k){
      if (aavex[k] < minex)  minex = aavex[k];
    }

    double maxex = 0;
    for(int k = 1; k < ar; ++k){
      if (aavex[k] > maxex)  maxex = aavex[k];
    }

    fprintf(fw, "%d %f  %f  %f\n", trial, avex, minex, maxex); 
    fflush(fw);

    printf(" [<avg> = %.3f <min> = %.3f <max> = %.3f]\n\n", avex, minex, maxex);
    printarrayfrag(aex, ar, "aex");
    printarrayfrag(aavex, ar, "aavex");
    printindexarrayfrag(aexE, arts, ar, "aexE");

    // update aT, R, ar after insertion
    insert_temps(aavex, aT, &R, &ar, ains);

    // update aT
    rebuild_temps(aT, R, ar);

    // update arts and atrs
    rebuild_indices(arts, atrs, ar);

  } // atrials

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel time (metropolis simulation) %.2f secs\n", total_ktime * 1e-9);
  printf("Total trial time %.2f secs\n", time * 1e-9);

  fclose(fw);
  for(int i = 0; i < rpool; ++i) {
    
    
    
  }

  
  
  
  free(T);
  free(aex);
  free(aavex);
  free(aexE);
  delete[] mdlat;
  delete[] apcga;
  delete[] apcgb;
  free(arts);
  free(atrs);
  free(aT);

  release_mem();
  platform.finalize();
  return 0;
}
