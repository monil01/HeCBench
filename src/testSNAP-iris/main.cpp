// ----------------------------------------------------------------------
// Copyright (2019) Sandia Corporation.
// Under the terms of Contract DE-AC04-94AL85000
// with Sandia Corporation, the U.S. Government
// retains certain rights in this software. This
// software is distributed under the Zero Clause
// BSD License
//
// TestSNAP - A prototype for the SNAP force kernel
// Version 0.0.2
// Main changes: Y array trick, memory compaction
//
// Original author: Aidan P. Thompson, athomps@sandia.gov
// http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
//
// ----------------------------------------------------------------------
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iris/iris.hpp>
#include "backend.h"
#include "snap.h"
#include "utils_host.h"
#include "launch.h"

#if REFDATA_TWOJ == 14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ == 8
#include "refdata_2J8_W.h"
#elif REFDATA_TWOJ == 4
#include "refdata_2J4_W.h"
#else
#include "refdata_2J2_W.h"
#endif


int nsteps = 1; // num of force evaluations

int main(int argc, char* argv[])
{
  iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
  options(argc, argv);

  const int switch_flag = 1;     // SNAP parameter

  // record timings of individual routines
  double elapsed_ui = 0.0, 
         elapsed_yi = 0.0, 
         elapsed_duidrj = 0.0,
         elapsed_deidrj = 0.0;

  const int ninside = refdata.ninside;
  const int ncoeff = refdata.ncoeff;
  const int nlocal = refdata.nlocal;
  const int nghost = refdata.nghost;
  const int ntotal = nlocal + nghost;
  const int twojmax = refdata.twojmax;
  const double rcutfac = refdata.rcutfac;

  const double wself = 1.0;
  const int num_atoms = nlocal; 
  const int num_nbor = ninside; 

  //coeffi = memory->grow(coeffi, ncoeff + 1, "coeffi");
  double* coeffi = (double*) malloc (sizeof(double) * (ncoeff+1));

  for (int icoeff = 0; icoeff < ncoeff + 1; icoeff++)
    coeffi[icoeff] = refdata.coeff[icoeff];

  double* beta = coeffi + 1;

  // build index list
  const int jdim = twojmax + 1;

  // index list for cglist

  int *idxcg_block = (int*) malloc(sizeof(int) * jdim * jdim * jdim);

  int idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxcg_block[j1 + j2 *jdim + jdim*jdim*j] = idxcg_count;
        for (int m1 = 0; m1 <= j1; m1++)
          for (int m2 = 0; m2 <= j2; m2++)
            idxcg_count++;
      }
  const int idxcg_max = idxcg_count;

  // index list for uarray
  // need to include both halves
  // **** only place rightside is used is in compute_yi() ***

  int* idxu_block = (int*) malloc (sizeof(int) * jdim);
  int idxu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    idxu_block[j] = idxu_count;
    for (int mb = 0; mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxu_count++;
  }
  const int idxu_max = idxu_count;

  // parity list for uarray inversion symmetry
  // parity +1: u[ma-j][mb-j] = +Conj([u[ma][mb])
  // parity -1: u[ma-j][mb-j] = -Conj([u[ma][mb])

  // ulist_parity.resize(idxu_max);
  int* ulist_parity = (int*) malloc (sizeof(int) * idxu_max);
  idxu_count = 0;
  for (int j = 0; j <= twojmax; j++) {
    int mbpar = 1;
    for (int mb = 0; mb <= j; mb++) {
      int mapar = mbpar;
      for (int ma = 0; ma <= j; ma++) {
        ulist_parity[idxu_count] = mapar;
        mapar = -mapar;
        idxu_count++;
      }
      mbpar = -mbpar;
    }
  }

  // index list for duarray, yarray
  // only include left half
  // NOTE: idxdu indicates lefthalf only
  //       idxu indicates both halves

  int* idxdu_block = (int*) malloc (sizeof(int) * jdim);
  int idxdu_count = 0;

  for (int j = 0; j <= twojmax; j++) {
    idxdu_block[j] = idxdu_count;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++)
        idxdu_count++;
  }
  const int idxdu_max = idxdu_count;

  // index list for beta and B

  int idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1)
          idxb_count++;

  const int idxb_max = idxb_count;
  SNA_BINDICES* idxb = (SNA_BINDICES*) malloc (sizeof(SNA_BINDICES) * idxb_max);

  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1) {
          idxb[idxb_count].j1 = j1;
          idxb[idxb_count].j2 = j2;
          idxb[idxb_count].j = j;
          idxb_count++;
        }

  // reverse index list for beta and b

  int* idxb_block = (int*) malloc (sizeof(int) * jdim * jdim * jdim);
  idxb_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        if (j < j1)
          continue;
        idxb_block[j1*jdim*jdim+j2*jdim+j] = idxb_count;
        idxb_count++;
      }


  // index list for zlist

  int idxz_count = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++)
            idxz_count++;

  const int idxz_max = idxz_count;
  //idxz.resize(idxz_max, 9);
  int* idxz = (int*) malloc (sizeof(int) * idxz_max * 9);

  //idxzbeta.resize(idxz_max);
  double* idxzbeta = (double*) malloc (sizeof(double) * idxz_max);

  //memory->create(idxz_block, jdim, jdim, jdim, "sna:idxz_block");
  int* idxz_block = (int*) malloc (sizeof(int) * jdim * jdim * jdim);

  idxz_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        idxz_block[j1*jdim*jdim+j2*jdim+j] = idxz_count;

        // find right beta[jjb] entries
        // multiply and divide by j+1 factors
        // account for multiplicity of 1, 2, or 3
        // this should not be computed here

        double betaj;
        if (j >= j1) {
          const int jjb = idxb_block[j1*jdim*jdim+j2*jdim+j];
          if (j1 == j) {
            if (j2 == j) {
              betaj = 3 * beta[jjb];
            }
            else {
              betaj = 2 * beta[jjb];
            }
          } else {
            betaj = beta[jjb];
          }
        } else if (j >= j2) {
          const int jjb = idxb_block[j*jdim*jdim+j2*jdim+j1];
          if (j2 == j) {
            betaj = 2 * beta[jjb] * (j1 + 1) / (j + 1.0);
          }
          else {
            betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
          }
        } else {
          const int jjb = idxb_block[j2*jdim*jdim+j*jdim+j1];
          betaj = beta[jjb] * (j1 + 1) / (j + 1.0);
        }

        for (int mb = 0; 2 * mb <= j; mb++)
          for (int ma = 0; ma <= j; ma++) {

            idxz[IDXZ_INDEX(idxz_count, 0)] = j1;
            idxz[IDXZ_INDEX(idxz_count, 1)] = j2;
            idxz[IDXZ_INDEX(idxz_count, 2)] = j;

            int ma1min = MAX(0, (2 * ma - j - j2 + j1) / 2);
            idxz[IDXZ_INDEX(idxz_count, 3)] = ma1min;
            idxz[IDXZ_INDEX(idxz_count, 4)] = (2 * ma - j - (2 * ma1min - j1) + j2) / 2;
            idxz[IDXZ_INDEX(idxz_count, 5)] =
              MIN(j1, (2 * ma - j + j2 + j1) / 2) - ma1min + 1;

            int mb1min = MAX(0, (2 * mb - j - j2 + j1) / 2);
            idxz[IDXZ_INDEX(idxz_count, 6)] = mb1min;
            idxz[IDXZ_INDEX(idxz_count, 7)] = (2 * mb - j - (2 * mb1min - j1) + j2) / 2;
            idxz[IDXZ_INDEX(idxz_count, 8)] =
              MIN(j1, (2 * mb - j + j2 + j1) / 2) - mb1min + 1;

            idxzbeta[idxz_count] = betaj;

            idxz_count++;
          }
      }
  // omit beta0 from beta vector


  if (compute_ncoeff(twojmax) != ncoeff) {
    printf("ERROR: ncoeff from SNA does not match reference data\n");
    exit(1);
  }

  //snaptr->grow_rij(ninside);

  double *rij    = (double*) malloc(sizeof(double) * (num_atoms * num_nbor * 3));
  double *inside = (double*) malloc(sizeof(double) * (num_atoms * num_nbor));
  double *wj     = (double*) malloc(sizeof(double) * (num_atoms * num_nbor));
  double *rcutij = (double*) malloc(sizeof(double) * (num_atoms * num_nbor));

  const int jdimpq = twojmax + 2;
  double* rootpqarray = (double*) malloc(sizeof(double) * jdimpq * jdimpq);
  double* cglist = (double*) malloc (sizeof(double) * idxcg_max);
  double* dedr = (double*) malloc (sizeof(double) * num_atoms * num_nbor * 3); 

  COMPLEX* ulist = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * num_nbor * idxu_max); 
  COMPLEX* ylist = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * idxdu_max);
  COMPLEX* ulisttot = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * idxu_max);
  COMPLEX* dulist = (COMPLEX*) malloc (sizeof(COMPLEX) * num_atoms * num_nbor * 3 * idxdu_max);

  // init rootpqarray
  for (int p = 1; p <= twojmax; p++)
    for (int q = 1; q <= twojmax; q++)
      rootpqarray[ROOTPQ_INDEX(p, q)] = sqrt(static_cast<double>(p) / q);

  // init_clebsch_gordan()
  double sum, dcg, sfaccg;
  int m, aa2, bb2, cc2;
  int ifac;

  idxcg_count = 0;
  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2) {
        for (int m1 = 0; m1 <= j1; m1++) {
          aa2 = 2 * m1 - j1;

          for (int m2 = 0; m2 <= j2; m2++) {

            // -c <= cc <= c

            bb2 = 2 * m2 - j2;
            m = (aa2 + bb2 + j) / 2;

            if (m < 0 || m > j) {
              cglist[idxcg_count] = 0.0;
              idxcg_count++;
              continue;
            }

            sum = 0.0;

            for (int z = MAX(0, MAX(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
                z <=
                MIN((j1 + j2 - j) / 2, MIN((j1 - aa2) / 2, (j2 + bb2) / 2));
                z++) {
              ifac = z % 2 ? -1 : 1;
              sum += ifac / (factorial(z) * factorial((j1 + j2 - j) / 2 - z) *
                  factorial((j1 - aa2) / 2 - z) *
                  factorial((j2 + bb2) / 2 - z) *
                  factorial((j - j2 + aa2) / 2 + z) *
                  factorial((j - j1 - bb2) / 2 + z));
            }

            cc2 = 2 * m - j;
            dcg = deltacg(j1, j2, j);
            sfaccg = sqrt(
                factorial((j1 + aa2) / 2) * factorial((j1 - aa2) / 2) *
                factorial((j2 + bb2) / 2) * factorial((j2 - bb2) / 2) *
                factorial((j + cc2) / 2) * factorial((j - cc2) / 2) * (j + 1));

            cglist[idxcg_count] = sum * dcg * sfaccg;
            idxcg_count++;
          }
        }
      }

  double* f = (double*) malloc (sizeof(double) * ntotal * 3);

  // initialize error tally
  double sumsqferr = 0.0;

  iris::DMem* d_idxu_block;
  d_idxu_block = new iris::DMem(idxu_block, sizeof(int)*jdim);
  d_idxu_block->update(idxu_block);

  iris::DMem* d_ulist_parity;
  d_ulist_parity = new iris::DMem(ulist_parity, sizeof(int)*idxu_max);
  d_ulist_parity->update(ulist_parity);

  iris::DMem* d_rootpqarray;
  d_rootpqarray = new iris::DMem(rootpqarray, sizeof(double)*jdimpq*jdimpq);
  d_rootpqarray->update(rootpqarray); 

  iris::DMem* d_idxz;
  d_idxz = new iris::DMem(idxz, sizeof(int)*idxz_max*9);
  d_idxz->update(idxz);

  iris::DMem* d_idxzbeta;
  d_idxzbeta = new iris::DMem(idxzbeta, sizeof(double)*idxz_max);
  d_idxzbeta->update(idxzbeta);

  iris::DMem* d_idxcg_block;
  d_idxcg_block = new iris::DMem(idxcg_block, sizeof(int)*jdim*jdim*jdim);
  d_idxcg_block->update(idxcg_block);

  iris::DMem* d_idxdu_block;
  d_idxdu_block = new iris::DMem(idxdu_block, sizeof(int)*jdim);
  d_idxdu_block->update(idxdu_block);

  iris::DMem* d_cglist;
  d_cglist = new iris::DMem(cglist, sizeof(double)*idxcg_max);
  d_cglist->update(cglist);

  iris::DMem* d_dulist;
  d_dulist = new iris::DMem(dulist, sizeof(COMPLEX)*num_atoms*num_nbor*3*idxdu_max);
  d_dulist->update(dulist);

  iris::DMem* d_ulist;
  d_ulist = new iris::DMem(ulist, sizeof(COMPLEX)*num_atoms*num_nbor*idxu_max);
  d_ulist->update(ulist);

  iris::DMem* d_dedr;
  d_dedr = new iris::DMem(dedr, sizeof(double)*num_atoms*num_nbor*3);
  d_dedr->update(dedr);

  iris::DMem* d_ulisttot;
  d_ulisttot = new iris::DMem(ulisttot, sizeof(COMPLEX)*num_atoms*idxu_max);

  iris::DMem* d_ylist;
  d_ylist = new iris::DMem(ylist, sizeof(COMPLEX)*num_atoms*idxdu_max);

  iris::DMem* d_rij;
  d_rij = new iris::DMem(rij, sizeof(double)*num_atoms*num_nbor*3);

  iris::DMem* d_rcutij;
  d_rcutij = new iris::DMem(rcutij, sizeof(double)*num_atoms*num_nbor);

  iris::DMem* d_wj;
  d_wj = new iris::DMem(wj, sizeof(double)*num_atoms*num_nbor);


  // loop over steps

  auto begin = myclock::now();

  for (int istep = 0; istep < nsteps; istep++) {

    time_point<system_clock> start, end;
    duration<double> elapsed;

    for (int j = 0; j < ntotal * 3; j++) {
      f[j] = 0.0;
    }

    int jt = 0, jjt = 0;
    for (int natom = 0; natom < num_atoms; natom++) {
      for (int nbor = 0; nbor < num_nbor; nbor++) {
        // n1 = num_atoms, n2 = num_nbor, n3 = 3
        rij[ULIST_INDEX(natom, nbor, 0)] = refdata.rij[jt++];
        rij[ULIST_INDEX(natom, nbor, 1)] = refdata.rij[jt++];
        rij[ULIST_INDEX(natom, nbor, 2)] = refdata.rij[jt++];
        inside[INDEX_2D(natom, nbor)] = refdata.jlist[jjt++];
        wj[INDEX_2D(natom, nbor)] = 1.0;
        rcutij[INDEX_2D(natom, nbor)] = rcutfac;
      }
    }

    d_rij->update(rij);
    d_rcutij->update(rcutij);
    d_wj->update(wj);

    // compute_ui
    start = system_clock::now();
    // compute_ui();
    // utot(j,ma,mb) = 0 for all j,ma,ma
    // utot(j,ma,ma) = 1 for all j,ma
    // for j in neighbors of i:
    //   compute r0 = (x,y,z,z0)
    //   utot(j,ma,mb) += u(r0;j,ma,mb) for all j,ma,mb

    Dim3 grid_k1 ((num_atoms*idxu_max+255)/256);
    Dim3 block_k1 (256);
    launch_reset_ulisttot(grid_k1, block_k1, d_ulisttot, num_atoms*idxu_max);


    Dim3 grid_k2 ((num_atoms+255)/256);
    Dim3 block_k2 (256);
    launch_set_ulisttot(grid_k2, block_k2, d_ulisttot, d_idxu_block, num_atoms, twojmax, wself);


    Dim3 grid_k3 ((num_atoms+15)/16, (num_nbor+15)/16);
    Dim3 block_k3 (16, 16);

    launch_update_ulisttot(grid_k3, block_k3, 
        d_rij, 
        d_rcutij,
        d_wj,
        d_ulist_parity,
        d_idxu_block,
        d_rootpqarray,
        d_ulist,
        d_ulisttot,
        num_atoms,
        num_nbor,
        switch_flag,
        twojmax,
        jdimpq);

    
    end = system_clock::now();
    elapsed = end - start;
    elapsed_ui += elapsed.count();

    start = system_clock::now();

    // Initialize ylist elements to zeros
    Dim3 grid_k4 ((num_atoms*idxdu_max+255)/256);
    Dim3 block_k4 (256);

    launch_reset_ylist(grid_k4, block_k4, d_ylist, num_atoms*idxdu_max);

    Dim3 grid_k5 ((num_atoms+15)/16, (idxz_max+15)/16);
    Dim3 block_k5 (16, 16);

    launch_compute_yi(grid_k5, block_k5, 
        d_idxz,
        d_idxzbeta,
        d_cglist,
        d_idxcg_block,
        d_idxu_block,
        d_idxdu_block,
        d_ulisttot,
        d_ylist,
        num_atoms,
        idxz_max,
        jdim);

    
    end = system_clock::now();
    elapsed = end - start;
    elapsed_yi += elapsed.count();

    // compute_duidrj
    start = system_clock::now();

    Dim3 grid_k6 ((num_atoms+15)/16, (num_nbor+15)/16);
    Dim3 block_k6 (16, 16);
    launch_compute_duidrj(grid_k6, block_k6, 
        d_wj,
        d_rij,
        d_rcutij,
        d_rootpqarray,
        d_ulist,
        d_dulist,
        num_atoms,
        num_nbor,
        twojmax,
        idxdu_max,
        jdimpq,
        switch_flag);

    
    end = system_clock::now();
    elapsed = end - start;
    elapsed_duidrj += elapsed.count();

    start = system_clock::now();

    // compute_deidrj();
    Dim3 grid_k7 ((num_atoms+15)/16, (num_nbor+15)/16);
    Dim3 block_k7 (16, 16);

    launch_compute_deidrj(grid_k7, block_k7,  
        d_idxdu_block,
        d_dulist,
        d_ylist,
        d_dedr,
        num_atoms,
        num_nbor,
        twojmax,
        idxdu_max);

    
    end = system_clock::now();
    elapsed = end - start;
    elapsed_deidrj += elapsed.count();

    d_dedr->fetch(dedr, sizeof(double)*num_atoms*num_nbor*3);

    // Compute forces and error
    //compute_forces(snaptr);
    for (int natom = 0; natom < num_atoms; natom++) {
      for (int nbor = 0; nbor < num_nbor; nbor++) {
        int j = inside[INDEX_2D(natom, nbor)];
        f[F_INDEX(natom, 0)] += dedr[ULIST_INDEX(natom, nbor, 0)];
        f[F_INDEX(natom, 1)] += dedr[ULIST_INDEX(natom, nbor, 1)];
        f[F_INDEX(natom, 2)] += dedr[ULIST_INDEX(natom, nbor, 2)];
        f[F_INDEX(j, 0)] -= dedr[ULIST_INDEX(natom, nbor, 0)];
        f[F_INDEX(j, 1)] -= dedr[ULIST_INDEX(natom, nbor, 1)];
        f[F_INDEX(j, 2)] -= dedr[ULIST_INDEX(natom, nbor, 2)];

      } // loop over neighbor forces
    }   // loop over atoms
    //    compute_error(snaptr);
    jt = 0;
    for (int j = 0; j < ntotal; j++) {
      double ferrx = f[F_INDEX(j, 0)] - refdata.fj[jt++];
      double ferry = f[F_INDEX(j, 1)] - refdata.fj[jt++];
      double ferrz = f[F_INDEX(j, 2)] - refdata.fj[jt++];
      sumsqferr += ferrx * ferrx + ferry * ferry + ferrz * ferrz;
    }
  }
  auto stop = myclock::now();
  myduration elapsed = stop - begin;
  double duration = elapsed.count(); 

  printf("-----------------------\n");
  printf("Summary of TestSNAP run\n");
  printf("-----------------------\n");
  printf("natoms = %d \n", nlocal);
  printf("nghostatoms = %d \n", nghost);
  printf("nsteps = %d \n", nsteps);
  printf("nneighs = %d \n", ninside);
  printf("twojmax = %d \n", twojmax);
  printf("duration = %g [sec]\n", duration);

  // step time includes host, device, and host-data transfer time
  double ktime = elapsed_ui + elapsed_yi + elapsed_duidrj + elapsed_deidrj;
  printf("step time = %g [msec/step]\n", 1000.0 * duration / nsteps);
  printf("\n Individual kernel timings for each step\n");
  printf("   compute_ui = %g [msec/step]\n", 1000.0 * elapsed_ui / nsteps);
  printf("   compute_yi = %g [msec/step]\n", 1000.0 * elapsed_yi / nsteps);
  printf("   compute_duidrj = %g [msec/step]\n", 1000.0 * elapsed_duidrj / nsteps);
  printf("   compute_deidrj = %g [msec/step]\n", 1000.0 * elapsed_deidrj / nsteps);
  printf("   Total kernel time = %g [msec/step]\n", 1000.0 * ktime / nsteps);
  printf("   Percentage of step time = %g%%\n\n", ktime / duration * 100.0);
  printf("grind time = %g [msec/atom-step]\n", 1000.0 * duration / (nlocal * nsteps));
  printf("RMS |Fj| deviation %g [eV/A]\n", sqrt(sumsqferr / (ntotal * nsteps)));

  delete d_idxu_block;
  delete d_ulist_parity;
  delete d_rootpqarray;
  delete d_idxz;
  delete d_idxzbeta;
  delete d_idxcg_block;
  delete d_idxdu_block;
  delete d_cglist;
  delete d_dulist;
  delete d_ulist;
  delete d_dedr;
  delete d_ulisttot;
  delete d_ylist;
  delete d_rij;
  delete d_rcutij;
  delete d_wj;

  free(coeffi);
  free(idxcg_block);
  free(idxu_block);
  free(ulist_parity);
  free(idxdu_block);
  free(idxb);
  free(idxb_block);
  free(idxz);
  free(idxzbeta);
  free(idxz_block);
  free(rij);
  free(inside);
  free(wj);
  free(rcutij);
  free(rootpqarray);
  free(cglist);
  free(dedr);
  free(ulist);
  free(ylist);
  free(ulisttot);
  free(dulist);
  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *out=fopen(path,"wb"); if(!out) exit(4);
    int meta[]={ntotal,nsteps,num_atoms};
    if(fwrite(meta,sizeof(int),3,out)!=3) exit(4);
    if(fwrite(f,sizeof(double),3*ntotal,out)!=size_t(3*ntotal)) exit(4);
    if(fwrite(refdata.fj,sizeof(double),3*ntotal,out)!=size_t(3*ntotal)) exit(4);
    fclose(out);
  }
  free(f);

  platform.finalize();
  return 0;
}
