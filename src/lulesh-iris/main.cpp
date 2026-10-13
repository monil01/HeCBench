/*
   Copyright (c) 2010-2013.
   Lawrence Livermore National Security, LLC.
   Produced at the Lawrence Livermore National Laboratory.
   LLNL-CODE-461231
   All rights reserved.

   This file is part of LULESH, Version 2.0.
   Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
 * Addition of regions to make work more representative of multi-material codes
 * Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
 more representative of our actual working set sizes
 * Single source distribution supports pure serial, pure OpenMP, MPI-only,
 and MPI+OpenMP
 * Addition of ability to visualize the mesh using VisIt
https://wci.llnl.gov/codes/visit/download.html
 * Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
 printf(" where [opts] is one or more of:\n");
 printf(" -q              : quiet mode - suppress all stdout\n");
 printf(" -i <iterations> : number of cycles to run\n");
 printf(" -s <size>       : length of cube mesh along side\n");
 printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
 printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
 printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
 printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
 printf(" -p              : Print out progress\n");
 printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
 printf(" -h              : This message\n");
 printf("\n\n");

 *Notable changes in LULESH 2.0

 * Split functionality into different files
 lulesh.cc - where most (all?) of the timed functionality lies
 lulesh-comm.cc - MPI functionality
 lulesh-init.cc - Setup code
 lulesh-viz.cc  - Support for visualization option
 lulesh-util.cc - Non-timed functions
 *
 * The concept of "regions" was added, although every region is the same ideal
 *    gas material, and the same sedov blast wave problem is still the only
 *    problem its hardcoded to solve.
 * Regions allow two things important to making this proxy app more representative:
 *   Four of the LULESH routines are now performed on a region-by-region basis,
 *     making the memory access patterns non-unit stride
 *   Artificial load imbalances can be easily introduced that could impact
 *     parallelization strategies.
 * The load balance flag changes region assignment.  Region number is raised to
 *   the power entered for assignment probability.  Most likely regions changes
 *   with MPI process id.
 * The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
 *   entered multiple. The cost of 5% is 10x the entered multiple.
 * MPI and OpenMP were added, and coalesced into a single version of the source
 *   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
  *
* Possible Differences in Final Release (other changes possible)
  *
  * High Level mesh structure to allow data structure transformations
  * Different default parameters
  * Minor code performance changes and cleanup

  TODO in future versions
  * Add reader for (truly) unstructured meshes, probably serial only
  * CMake based build system

  //////////////

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

  * Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the disclaimer (as noted below)
  in the documentation and/or other materials provided with the
  distribution.

  * Neither the name of the LLNS/LLNL nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
  THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
      BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
      DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
      NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


  Additional BSD Notice

  1. This notice is required to be provided under our contract with the U.S.
  Department of Energy (DOE). This work was produced at Lawrence Livermore
  National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

  2. Neither the United States Government nor Lawrence Livermore National
  Security, LLC nor any of their employees, makes any warranty, express
  or implied, or assumes any liability or responsibility for the accuracy,
  completeness, or usefulness of any information, apparatus, product, or
  process disclosed, or represents that its use would not infringe
  privately-owned rights.

  3. Also, reference herein to any specific commercial products, process, or
  services by trade name, trademark, manufacturer or otherwise does not
  necessarily constitute or imply its endorsement, recommendation, or
  favoring by the United States Government or Lawrence Livermore National
  Security, LLC. The views and opinions of authors expressed herein do not
  necessarily state or reflect those of the United States Government or
  Lawrence Livermore National Security, LLC, and shall not be used for
  advertising or product endorsement purposes.

  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <sstream>
#include <limits>
#include <fstream>
#include <string>
#ifdef VERIFY
#include <cassert>
#endif
#include "lulesh.h"

#include "geometry.h"
#include "launch.h"
#include "backend.h"
/*********************************/
/* Data structure implementation */
/*********************************/

/* might want to add access methods so that memory can be */
/* better managed, as in luleshFT */

template <typename T>
T *Allocate(size_t size)
{
  return static_cast<T *>(malloc(sizeof(T)*size)) ;
}

template <typename T>
void Release(T **ptr)
{
  if (*ptr != NULL) {
    free(*ptr) ;
    *ptr = NULL ;
  }
}

/******************************************/

/* Work Routines */

static inline
void TimeIncrement(Domain& domain)
{
  Real_t targetdt = domain.stoptime() - domain.time() ;

  if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
    Real_t ratio ;
    Real_t olddt = domain.deltatime() ;

    /* This will require a reduction in parallel */
    Real_t gnewdt = Real_t(1.0e+20) ;
    Real_t newdt ;
    if (domain.dtcourant() < gnewdt) {
      gnewdt = domain.dtcourant() / Real_t(2.0) ;
    }
    if (domain.dthydro() < gnewdt) {
      gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
    }

    newdt = gnewdt;

    ratio = newdt / olddt ;
    if (ratio >= Real_t(1.0)) {
      if (ratio < domain.deltatimemultlb()) {
        newdt = olddt ;
      }
      else if (ratio > domain.deltatimemultub()) {
        newdt = olddt*domain.deltatimemultub() ;
      }
    }

    if (newdt > domain.dtmax()) {
      newdt = domain.dtmax() ;
    }
    domain.deltatime() = newdt ;
  }

  /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
  if ((targetdt > domain.deltatime()) &&
      (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
    targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
  }

  if (targetdt < domain.deltatime()) {
    domain.deltatime() = targetdt ;
  }

  domain.time() += domain.deltatime() ;

  ++domain.cycle() ;
}


/******************************************/


/******************************************/

static inline
void CalcCourantConstraintForElems(Domain &domain, Index_t length,
    Index_t *regElemlist,
    Real_t qqc, Real_t& dtcourant)
{
  Index_t threads = NT;
  static Index_t *courant_elem_per_thread;
  static Real_t *dtcourant_per_thread;
  static bool first = true;
  if (first) {
    courant_elem_per_thread = new Index_t[threads];
    dtcourant_per_thread = new Real_t[threads];
    first = false;
  }

  //#pragma omp parallel firstprivate(length, qqc)
  {
    Real_t   qqc2 = Real_t(64.0) * qqc * qqc ;
    Real_t   dtcourant_tmp = dtcourant;
    Index_t  courant_elem  = -1 ;

    for (Index_t thread_num = 0; thread_num < NT; thread_num++) {

      //#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
        Index_t indx = regElemlist[i] ;
        Real_t dtf = domain.ss(indx) * domain.ss(indx) ;

        if ( domain.vdov(indx) < Real_t(0.) ) {
          dtf = dtf
            + qqc2 * domain.arealg(indx) * domain.arealg(indx)
            * domain.vdov(indx) * domain.vdov(indx) ;
        }

        dtf = SQRT(dtf) ;
        dtf = domain.arealg(indx) / dtf ;

        if (domain.vdov(indx) != Real_t(0.)) {
          if ( dtf < dtcourant_tmp ) {
            dtcourant_tmp = dtf ;
            courant_elem  = indx ;
          }
        }
      }

      dtcourant_per_thread[thread_num]    = dtcourant_tmp ;
      courant_elem_per_thread[thread_num] = courant_elem ;
    }
  }

  for (Index_t i = 1; i < threads; ++i) {
    if (dtcourant_per_thread[i] < dtcourant_per_thread[0] ) {
      dtcourant_per_thread[0]    = dtcourant_per_thread[i];
      courant_elem_per_thread[0] = courant_elem_per_thread[i];
    }
  }

  if (courant_elem_per_thread[0] != -1) {
    dtcourant = dtcourant_per_thread[0] ;
  }

  return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &domain, Index_t length,
    Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
{
  Index_t threads = NT;
  static Index_t *hydro_elem_per_thread;
  static Real_t *dthydro_per_thread;
  static bool first = true;
  if (first) {
    hydro_elem_per_thread = new Index_t[threads];
    dthydro_per_thread = new Real_t[threads];
    first = false;
  }

  //#pragma omp parallel firstprivate(length, dvovmax)
  {
    Real_t dthydro_tmp = dthydro ;
    Index_t hydro_elem = -1 ;

    for (Index_t thread_num = 0; thread_num < NT; thread_num++) {

      //#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
        Index_t indx = regElemlist[i] ;

        if (domain.vdov(indx) != Real_t(0.)) {
          Real_t dtdvov = dvovmax / (FABS(domain.vdov(indx))+Real_t(1.e-20)) ;

          if ( dthydro_tmp > dtdvov ) {
            dthydro_tmp = dtdvov ;
            hydro_elem = indx ;
          }
        }
      }

      dthydro_per_thread[thread_num]    = dthydro_tmp ;
      hydro_elem_per_thread[thread_num] = hydro_elem ;
    }
  }

  for (Index_t i = 1; i < threads; ++i) {
    if(dthydro_per_thread[i] < dthydro_per_thread[0]) {
      dthydro_per_thread[0]    = dthydro_per_thread[i];
      hydro_elem_per_thread[0] =  hydro_elem_per_thread[i];
    }
  }

  if (hydro_elem_per_thread[0] != -1) {
    dthydro =  dthydro_per_thread[0] ;
  }

  return ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems(Domain& domain) {

  // Initialize conditions to a very large value
  domain.dtcourant() = 1.0e+20;
  domain.dthydro() = 1.0e+20;

  for (Index_t r=0 ; r < domain.numReg() ; ++r) {
    /* evaluate time constraint */
    CalcCourantConstraintForElems(domain, domain.regElemSize(r),
        domain.regElemlist(r),
        domain.qqc(),
        domain.dtcourant()) ;

    /* check hydro constraint */
    CalcHydroConstraintForElems(domain, domain.regElemSize(r),
        domain.regElemlist(r),
        domain.dvovmax(),
        domain.dthydro()) ;
  }
}
/******************************************/


/******************************************/

int main(int argc, char *argv[])
{
  iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
  Domain *locDom ;
  Int_t numRanks ;
  Int_t myRank ;
  struct cmdLineOpts opts;

  numRanks = 1;
  myRank = 0;

  /* Set defaults that can be overridden by command line opts */
  opts.its = 9999999;
  opts.nx  = 30;
  opts.numReg = 11;
  opts.numFiles = (int)(numRanks+10)/9;
  opts.showProg = 0;
  opts.quiet = 0;
  opts.viz = 0;
  opts.balance = 1;
  opts.cost = 1;
  opts.iteration_cap = 0;

  ParseCommandLineOptions(argc, argv, myRank, &opts);

  if ((myRank == 0) && (opts.quiet == 0)) {
    printf("Running problem size %d^3 per domain until completion\n", opts.nx);
    printf("Num processors: %d\n", numRanks);
    printf("Num threads (hardcoded): %d\n", NT);
    printf("Total number of elements: %lld\n\n", (long long int)(numRanks*opts.nx*opts.nx*opts.nx));
    printf("To run other sizes, use -s <integer>.\n");
    printf("To run a fixed number of iterations, use -i <integer>.\n");
    printf("To run a more or less balanced region set, use -b <integer>.\n");
    printf("To change the relative costs of regions, use -c <integer>.\n");
    printf("To print out progress, use -p\n");
    printf("To write an output file for VisIt, use -v\n");
    printf("To only execute the first iteration, use -z (used when profiling: nvprof --metrics all)\n");
    printf("See help (-h) for more options\n\n");
  }

  // Set up the mesh and decompose. Assumes regular cubes for now
  Int_t col, row, plane, side;
  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

  // Build the main data structure and initialize it
  locDom = new Domain(numRanks, col, row, plane, opts.nx,
      side, opts.numReg, opts.balance, opts.cost) ;

  // BEGIN timestep to solution */
  auto start = std::chrono::steady_clock::now();

  // Compute elem to reglist correspondence
  Index_t k = 0;
  for (Int_t r=0 ; r<locDom->numReg() ; r++) {
    Index_t numElemReg = locDom->regElemSize(r);
    Index_t *regElemList = locDom->regElemlist(r);
    Index_t rep;
    //Determine load imbalance for this region
    //round down the number with lowest cost
    if(r < locDom->numReg()/2)
      rep = 1;
    //you don't get an expensive region unless you at least have 5 regions
    else if(r < (locDom->numReg() - (locDom->numReg()+15)/20))
      rep = 1 + locDom->cost();
    //very expensive regions
    else
      rep = 10 * (1+ locDom->cost());
    // std::cout << "Elems: " << numElemReg << " Reps: " << rep << "\n";
    for (Index_t e=0 ; e<numElemReg ; e++){
      locDom->m_elemRep[regElemList[e]] = rep;
      locDom->m_elemElem[k] = regElemList[e];
      k++;
    }
  }

  //export persistent data to GPU
  Index_t numNode = locDom->numNode();
  Index_t numElem = locDom->numElem() ;
  Index_t numElem8 = numElem * 8;

  printf("numNode=%d numElem=%d\n", numNode, numElem);

#ifdef VERIFY
  assert(numElem > 0);
#endif

  Real_t *x = &locDom->m_x[0];
  Real_t *y = &locDom->m_y[0];
  Real_t *z = &locDom->m_z[0];
  Real_t *fx = &locDom->m_fx[0];
  Real_t *fy = &locDom->m_fy[0];
  Real_t *fz = &locDom->m_fz[0];
  Real_t *xd = &locDom->m_xd[0];
  Real_t *yd = &locDom->m_yd[0];
  Real_t *zd = &locDom->m_zd[0];
  Real_t *xdd = &locDom->m_xdd[0];
  Real_t *ydd = &locDom->m_ydd[0];
  Real_t *zdd = &locDom->m_zdd[0];

  Index_t *nodelist = &locDom->m_nodelist[0];

  Buffer *d_x;
  d_x = new Buffer(sizeof(Real_t)*numNode);
  d_x->update(x,sizeof(Real_t)*numNode);
  Buffer *d_y;
  d_y = new Buffer(sizeof(Real_t)*numNode);
  d_y->update(y,sizeof(Real_t)*numNode);
  Buffer *d_z;
  d_z = new Buffer(sizeof(Real_t)*numNode);
  d_z->update(z,sizeof(Real_t)*numNode);
  Buffer *d_fx;
  d_fx = new Buffer(sizeof(Real_t)*numNode);
  d_fx->update(fx,sizeof(Real_t)*numNode);
  Buffer *d_fy;
  d_fy = new Buffer(sizeof(Real_t)*numNode);
  d_fy->update(fy,sizeof(Real_t)*numNode);
  Buffer *d_fz;
  d_fz = new Buffer(sizeof(Real_t)*numNode);
  d_fz->update(fz,sizeof(Real_t)*numNode);
  Buffer *d_xd;
  d_xd = new Buffer(sizeof(Real_t)*numNode);
  d_xd->update(xd,sizeof(Real_t)*numNode);
  Buffer *d_yd;
  d_yd = new Buffer(sizeof(Real_t)*numNode);
  d_yd->update(yd,sizeof(Real_t)*numNode);
  Buffer *d_zd;
  d_zd = new Buffer(sizeof(Real_t)*numNode);
  d_zd->update(zd,sizeof(Real_t)*numNode);
  Buffer *d_xdd;
  d_xdd = new Buffer(sizeof(Real_t)*numNode);
  d_xdd->update(xdd,sizeof(Real_t)*numNode);
  Buffer *d_ydd;
  d_ydd = new Buffer(sizeof(Real_t)*numNode);
  d_ydd->update(ydd,sizeof(Real_t)*numNode);
  Buffer *d_zdd;
  d_zdd = new Buffer(sizeof(Real_t)*numNode);
  d_zdd->update(zdd,sizeof(Real_t)*numNode);
  Buffer *d_nodelist;
  d_nodelist = new Buffer(sizeof(Index_t)*numElem8);
  d_nodelist->update(nodelist,sizeof(Index_t)*numElem8);

  // factor out the buffers from the loop
  Buffer *d_determ;
  d_determ = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_fx_elem;
  d_fx_elem = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_fy_elem;
  d_fy_elem = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_fz_elem;
  d_fz_elem = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_dvdx;
  d_dvdx = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_dvdy;
  d_dvdy = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_dvdz;
  d_dvdz = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_x8n;
  d_x8n = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_y8n;
  d_y8n = new Buffer(sizeof(Real_t)*numElem8);
  Buffer *d_z8n;
  d_z8n = new Buffer(sizeof(Real_t)*numElem8);

  Buffer *d_sigxx;
  d_sigxx = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_sigyy;
  d_sigyy = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_sigzz;
  d_sigzz = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delv_xi;
  d_delv_xi = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delx_xi;
  d_delx_xi = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delv_eta;
  d_delv_eta = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delx_eta;
  d_delx_eta = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delv_zeta;
  d_delv_zeta = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delx_zeta;
  d_delx_zeta = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_p;
  d_p = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_q;
  d_q = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_volo;
  d_volo = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_v;
  d_v = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_vol_error;
  d_vol_error = new Buffer(sizeof(int));

  Index_t *nodeElemStart = &locDom->m_nodeElemStart[0];
  Index_t len1 = numNode + 1;
  Index_t *nodeElemCornerList = &locDom->m_nodeElemCornerList[0];
  Index_t len2 = nodeElemStart[numNode];

  Buffer *d_nodeElemStart;
  d_nodeElemStart = new Buffer(sizeof(Index_t)*len1);
  d_nodeElemStart->update(nodeElemStart,sizeof(Index_t)*len1);

  Buffer *d_nodeElemCornerList;
  d_nodeElemCornerList = new Buffer(sizeof(Index_t)*len2);
  d_nodeElemCornerList->update(nodeElemCornerList,sizeof(Index_t)*len2);
  Real_t  gamma[32] __attribute__((__aligned__(64)));
  gamma[0] = Real_t( 1.);
  gamma[1] = Real_t( 1.);
  gamma[2] = Real_t(-1.);
  gamma[3] = Real_t(-1.);
  gamma[4] = Real_t(-1.);
  gamma[5] = Real_t(-1.);
  gamma[6] = Real_t( 1.);
  gamma[7] = Real_t( 1.);
  gamma[8] = Real_t( 1.);
  gamma[9] = Real_t(-1.);
  gamma[10] = Real_t(-1.);
  gamma[11] = Real_t( 1.);
  gamma[12] = Real_t(-1.);
  gamma[13] = Real_t( 1.);
  gamma[14] = Real_t( 1.);
  gamma[15] = Real_t(-1.);
  gamma[16] = Real_t( 1.);
  gamma[17] = Real_t(-1.);
  gamma[18] = Real_t( 1.);
  gamma[19] = Real_t(-1.);
  gamma[20] = Real_t( 1.);
  gamma[21] = Real_t(-1.);
  gamma[22] = Real_t( 1.);
  gamma[23] = Real_t(-1.);
  gamma[24] = Real_t(-1.);
  gamma[25] = Real_t( 1.);
  gamma[26] = Real_t(-1.);
  gamma[27] = Real_t( 1.);
  gamma[28] = Real_t( 1.);
  gamma[29] = Real_t(-1.);
  gamma[30] = Real_t( 1.);
  gamma[31] = Real_t(-1.);

  Buffer *d_gamma;
  d_gamma = new Buffer(sizeof(Real_t)*32);
  d_gamma->update(gamma,32*sizeof(Real_t));

  Buffer *d_ss;
  d_ss = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_elemMass;
  d_elemMass = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_nodalMass;
  d_nodalMass = new Buffer(sizeof(Real_t)*numNode);

  Index_t size = locDom->sizeX();
  Index_t numNodeBC = (size+1)*(size+1) ;

  Buffer *d_symmX;
  d_symmX = new Buffer(sizeof(Index_t)*numNodeBC);
  Buffer *d_symmY;
  d_symmY = new Buffer(sizeof(Index_t)*numNodeBC);
  Buffer *d_symmZ;
  d_symmZ = new Buffer(sizeof(Index_t)*numNodeBC);

  Buffer *d_vdov;
  d_vdov = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_delv;
  d_delv = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_arealg;
  d_arealg = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_dxx;
  d_dxx = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_dyy;
  d_dyy = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_dzz;
  d_dzz = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_vnew;
  d_vnew = new Buffer(sizeof(Real_t)*numElem);

  Buffer *d_lzetam;
  d_lzetam = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_lzetap;
  d_lzetap = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_letap;
  d_letap = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_letam;
  d_letam = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_lxip;
  d_lxip = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_lxim;
  d_lxim = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_elemBC;
  d_elemBC = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_ql;
  d_ql = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_qq;
  d_qq = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_e;
  d_e = new Buffer(sizeof(Real_t)*numElem);
  Buffer *d_elemRep;
  d_elemRep = new Buffer(sizeof(Index_t)*numElem);
  Buffer *d_elemElem;
  d_elemElem = new Buffer(sizeof(Index_t)*numElem);

  // Sum contributions to total stress tensor
  Real_t *p = &locDom->m_p[0];
  Real_t *q = &locDom->m_q[0];

  d_p->update(p,sizeof(Real_t)*numElem);
  d_q->update(q,sizeof(Real_t)*numElem);

  Real_t *volo = &locDom->m_volo[0];
  d_volo->update(volo,sizeof(Real_t)*numElem);

  Real_t *nodalMass = &locDom->m_nodalMass[0];
  d_nodalMass->update(nodalMass,sizeof(Real_t)*numNode);

  Index_t *symmX = &locDom->m_symmX[0];
  Index_t *symmY = &locDom->m_symmY[0];
  Index_t *symmZ = &locDom->m_symmZ[0];

  d_symmX->update(symmX,sizeof(Index_t)*numNodeBC);
  d_symmY->update(symmY,sizeof(Index_t)*numNodeBC);
  d_symmZ->update(symmZ,sizeof(Index_t)*numNodeBC);

  Real_t *delv = &locDom->m_delv[0];
  d_delv->update(delv,sizeof(Real_t)*numElem);

  Real_t *arealg = &locDom->m_arealg[0];
  d_arealg->update(arealg,sizeof(Real_t)*numElem);

  Index_t *elemBC = &locDom->m_elemBC[0];
  Index_t *lxim = &locDom->m_lxim[0];
  Index_t *lxip = &locDom->m_lxip[0];
  Index_t *letam = &locDom->m_letam[0];
  Index_t *letap = &locDom->m_letap[0];
  Index_t *lzetam = &locDom->m_lzetam[0];
  Index_t *lzetap = &locDom->m_lzetap[0];
  Real_t *elemMass = &locDom->m_elemMass[0];
  //Real_t *ql = &locDom->m_ql[0];
  //Real_t *qq = &locDom->m_qq[0];

  d_lzetam->update(lzetam,sizeof(Index_t)*numElem);
  d_lzetap->update(lzetap,sizeof(Index_t)*numElem);
  d_letam->update(letam,sizeof(Index_t)*numElem);
  d_letap->update(letap,sizeof(Index_t)*numElem);
  d_lxip->update(lxip,sizeof(Index_t)*numElem);
  d_lxim->update(lxim,sizeof(Index_t)*numElem);
  d_elemBC->update(elemBC,sizeof(Index_t)*numElem);
  d_elemMass->update(elemMass,sizeof(Real_t)*numElem);

  Index_t *elemRep = &locDom->m_elemRep[0];
  Index_t *elemElem = &locDom->m_elemElem[0];

  d_elemRep->update(elemRep,sizeof(Index_t)*numElem);
  d_elemElem->update(elemElem,sizeof(Index_t)*numElem);

  Real_t *v = &locDom->m_v[0];
  d_v->update(v,sizeof(Real_t)*numElem);

  Real_t *vdov = &locDom->m_vdov[0];
  d_vdov->update(vdov,sizeof(Real_t)*numElem);

  Real_t *e = &locDom->m_e[0];
  Real_t *ss = &locDom->m_ss[0];

  d_e->update(e,sizeof(Real_t)*numElem);
  d_ss->update(ss,sizeof(Real_t)*numElem);

  // error checking on the host
  Real_t *determ = Allocate<Real_t>(numElem) ;

  // resize m_dxx, m_dyy, and m_dzz
  locDom->AllocateStrains(numElem);

  Real_t *dxx = &locDom->m_dxx[0];
  Real_t *dyy = &locDom->m_dyy[0];
  Real_t *dzz = &locDom->m_dzz[0];

  d_dxx->update(dxx,sizeof(Real_t)*numElem);
  d_dyy->update(dyy,sizeof(Real_t)*numElem);
  d_dzz->update(dzz,sizeof(Real_t)*numElem);

  // resize position and velocity gradients
  Int_t allElem = numElem +  /* local elem */
      2*locDom->sizeX()*locDom->sizeY() + /* plane ghosts */
      2*locDom->sizeX()*locDom->sizeZ() + /* row ghosts */
      2*locDom->sizeY()*locDom->sizeZ() ; /* col ghosts */

  locDom->AllocateGradients(numElem, allElem);

  while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {

    TimeIncrement(*locDom) ;
    //==============================================================================
    // LagrangeLeapFrog(*locDom) ;
    //==============================================================================

    //=============================================================
    // calculate nodal forces, accelerations, velocities, positions, with
    // applied boundary conditions and slide surface considerations
    //LagrangeNodal(domain);
    //=============================================================

    Domain &domain = *locDom;

    const Real_t deltaTime = domain.deltatime() ;
    Real_t u_cut = domain.u_cut() ;

    //=============================================================================
    // time of boundary condition evaluation is beginning of step for force and
    // acceleration boundary conditions.
    //CalcForceForNodes(domain);
    //=============================================================================

    //=====================================================================
    // CalcVolumeForceForElems(domain)
    //=====================================================================

    Real_t  hgcoef = domain.hgcoef() ;

    Dim3 gws_elem ((numElem+THREADS-1)/THREADS);
    Dim3 gws_node ((numNode+THREADS-1)/THREADS);
    Dim3 lws (THREADS);

    launch_fill_sig(gws_elem, lws, d_sigxx, d_sigyy, d_sigzz, d_p, d_q, numElem);

    //==============================================================================================
    // IntegrateStressForElems( domain, sigxx, sigyy, sigzz, determ, numElem, domain.numNode())
    //==============================================================================================

    launch_integrateStress(gws_elem, lws, 
        d_fx_elem,
        d_fy_elem,
        d_fz_elem,
        d_x,
        d_y,
        d_z,
        d_nodelist,
        d_sigxx,
        d_sigyy,
        d_sigzz,
        d_determ,
        numElem);

    launch_acc_final_force(gws_node, lws, 
        d_fx_elem,
        d_fy_elem,
        d_fz_elem,
        d_fx,
        d_fy,
        d_fz,
        d_nodeElemStart,
        d_nodeElemCornerList,
        numNode);

#ifdef VERIFY
    // check for negative element volume on the host
    d_determ->read(determ,sizeof(Real_t)*numElem);

#ifdef _OPENMP
#pragma omp parallel for firstprivate(numElem)
#endif
    for ( Index_t k=0 ; k<numElem ; ++k ) {
      if (determ[k] <= Real_t(0.0)) {
        printf("determError: negative determ value @%d %f\n", k, determ[k]);
        //exit(VolumeError);
      }
    }
#endif

    //=================================================================================
    // CalcHourglassControlForElems(device_queue, domain, determ, hgcoef) ;
    //=================================================================================

    int vol_error = -1;

    d_vol_error->update(&vol_error,sizeof(int));

    launch_hgc(gws_elem, lws, 
        d_dvdx,
        d_dvdy,
        d_dvdz,
        d_x8n,
        d_y8n,
        d_z8n,
        d_determ,
        d_x,
        d_y,
        d_z,
        d_nodelist,
        d_volo,
        d_v,
        d_vol_error,
        numElem );

#ifdef DEBUG
    Real_t *dvdx_tmp = (Real_t*) malloc (sizeof(Real_t)*numElem8);
    Real_t *dvdy_tmp = (Real_t*) malloc (sizeof(Real_t)*numElem8);
    Real_t *dvdz_tmp = (Real_t*) malloc (sizeof(Real_t)*numElem8);
    Real_t *x8n_tmp = (Real_t*) malloc(sizeof(Real_t)*numElem8);
    Real_t *y8n_tmp = (Real_t*) malloc(sizeof(Real_t)*numElem8);
    Real_t *z8n_tmp = (Real_t*) malloc(sizeof(Real_t)*numElem8);

    d_dvdx->read(dvdx_tmp,sizeof(Real_t)*numElem8);
    d_dvdy->read(dvdy_tmp,sizeof(Real_t)*numElem8);
    d_dvdz->read(dvdz_tmp,sizeof(Real_t)*numElem8);
    d_x8n->read(x8n_tmp,sizeof(Real_t)*numElem8);
    d_y8n->read(y8n_tmp,sizeof(Real_t)*numElem8);
    d_z8n->read(z8n_tmp,sizeof(Real_t)*numElem8);
    d_determ->read(determ,sizeof(Real_t)*numElem);

    // volumn derivative
    for (int i = 0; i < numElem8; i++) {
      printf("vd %d %f %f %f %f %f %f %f\n",
          i, dvdx_tmp[i], dvdy_tmp[i], dvdz_tmp[i],
          x8n_tmp[i], y8n_tmp[i], z8n_tmp[i], determ[i/8]);
    }

    free(dvdx_tmp);
    free(dvdy_tmp);
    free(dvdz_tmp);
    free(x8n_tmp);
    free(y8n_tmp);
    free(z8n_tmp);
#endif

#ifdef VERIFY
    d_vol_error->read(&vol_error,sizeof(int));

    if (vol_error >= 0){
      printf("VolumeError: negative volumn\n");
      exit(VolumeError);
    }
#endif

    if ( hgcoef > Real_t(0.) ) {

      //Index_t *nodeElemStart = &domain.m_nodeElemStart[0];
      //Index_t len1 = numNode + 1;
      //Index_t *nodeElemCornerList = &domain.m_nodeElemCornerList[0];
      //Index_t len2 = nodeElemStart[numNode];

      launch_fb(gws_elem, lws, 
          d_dvdx,
          d_dvdy,
          d_dvdz,
          d_x8n,
          d_y8n,
          d_z8n,
          d_determ,
          d_xd,
          d_yd,
          d_zd,
          d_ss,
          d_elemMass,
          d_nodelist,
          d_gamma,
          d_fx_elem,
          d_fy_elem,
          d_fz_elem,
          hgcoef,
          numElem );

      launch_collect_final_force(gws_node, lws, 
          d_fx_elem,
          d_fy_elem,
          d_fz_elem,
          d_fx,
          d_fy,
          d_fz,
          d_nodeElemStart,
          d_nodeElemCornerList,
          numNode );

#ifdef DEBUG
      Real_t *fx_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
      Real_t *fy_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
      Real_t *fz_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);

      d_fx->read(fx_tmp,sizeof(Real_t)*numNode);
      d_fy->read(fy_tmp,sizeof(Real_t)*numNode);
      d_fz->read(fz_tmp,sizeof(Real_t)*numNode);

      for (int i = 0; i < numNode; i++)
        printf("fb: %d %f %f %f\n", i, fx_tmp[i], fy_tmp[i], fz_tmp[i]);

      free(fx_tmp);
      free(fy_tmp);
      free(fz_tmp);
#endif
    } // if ( hgcoef > Real_t(0.) )

    //===========================================================================
    //CalcAccelerationForNodes(domain, domain.numNode());   // IN: fx  OUT: m_xdd
    //===========================================================================

    launch_accelerationForNode(gws_node, lws, 
        d_fx,
        d_fy,
        d_fz,
        d_nodalMass,
        d_xdd,
        d_ydd,
        d_zdd,
        numNode);

    //======================================================================================
    //ApplyAccelerationBoundaryConditionsForNodes(domain); // uses m_xdd
    //======================================================================================
    //Index_t size = domain.sizeX();
    //Index_t numNodeBC = (size+1)*(size+1) ;

    Index_t s1 = domain.symmXempty();
    Index_t s2 = domain.symmYempty();
    Index_t s3 = domain.symmZempty();

    launch_applyAccelerationBoundaryConditionsForNodes((numNodeBC+255)/256, 256, 
        d_symmX,
        d_symmY,
        d_symmZ,
        d_xdd,
        d_ydd,
        d_zdd,
        s1,
        s2,
        s3,
        numNodeBC ) ;

    //=================================================================
    // CalcVelocityForNodes( domain, delt, u_cut, domain.numNode()) ; //uses m_xd and m_xdd
    //=================================================================
    launch_calcVelocityForNodes(gws_node, lws, 
        d_xd,
        d_yd,
        d_zd,
        d_xdd,
        d_ydd,
        d_zdd,
        deltaTime,
        u_cut,
        numNode );

    //=================================================================================
    // CalcPositionForNodes( domain, delt, domain.numNode() );  //uses m_xd and m_x
    //=================================================================================
    launch_calcPositionForNodes(gws_node, lws, 
        d_x,
        d_y,
        d_z,
        d_xd,
        d_yd,
        d_zd,
        deltaTime,
        numNode) ;

#ifdef DEBUG
    Real_t *xd_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
    Real_t *yd_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
    Real_t *zd_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
    Real_t *x_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
    Real_t *y_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);
    Real_t *z_tmp = (Real_t*) malloc (sizeof(Real_t)*numNode);

    d_xd->read(xd_tmp,sizeof(Real_t)*numNode);
    d_yd->read(yd_tmp,sizeof(Real_t)*numNode);
    d_zd->read(zd_tmp,sizeof(Real_t)*numNode);
    d_x->read(x_tmp,sizeof(Real_t)*numNode);
    d_y->read(y_tmp,sizeof(Real_t)*numNode);
    d_z->read(z_tmp,sizeof(Real_t)*numNode);

    for (int i = 0; i < numNode; i++)
      printf("CalcPositionForNodes: %d %f %f %f %f %f %f\n",
          i, x_tmp[i], y_tmp[i], z_tmp[i], xd_tmp[i], yd_tmp[i], zd_tmp[i]);

    free(x_tmp);
    free(y_tmp);
    free(z_tmp);
    free(xd_tmp);
    free(yd_tmp);
    free(zd_tmp);
#endif

    //=========================================================
    // calculate element quantities (i.e. velocity gradient & q), and update material states
    // LagrangeElements(domain);
    //=========================================================



    //========================================================================
    // void CalcKinematicsForElems( Domain &domain, Real_t *vnew,
    //========================================================================
    launch_calcKinematicsForElems(gws_elem, lws, 
        d_xd,
        d_yd,
        d_zd,
        d_x,
        d_y,
        d_z,
        d_nodelist,
        d_volo,
        d_v,
        d_delv,
        d_arealg,
        d_dxx,
        d_dyy,
        d_dzz,
        d_vnew,
        deltaTime,
        numElem );

    vol_error = -1; // reset volumn error

    d_vol_error->update(&vol_error,sizeof(int));

    launch_calcStrainRates(gws_elem, lws, 
        d_dxx,
        d_dyy,
        d_dzz,
        d_vnew,
        d_vdov,
        d_vol_error,
        numElem );

#ifdef VERIFY
    d_vol_error->read(&vol_error,sizeof(int));
    if (vol_error >= 0){
      printf("VolumeError: negative volumn\n");
      //exit(VolumeError);
    }
#endif

    d_vdov->read(vdov,sizeof(Real_t)*numElem);

#ifdef DEBUG
    for ( Index_t k=0 ; k<numElem ; ++k )
    {
      printf("kintec: %d %f\n", k, vdov[k]);
    }
#endif

    //=======================================================
    //CalcQForElems(domain, vnew) ;
    //=======================================================

    //================================================================
    // Calculate velocity gradients
    //CalcMonotonicQGradientsForElems(domain, vnew);
    //================================================================
    launch_calcMonotonicQGradientsForElems(gws_elem, lws, 
        d_xd,
        d_yd,
        d_zd,
        d_x,
        d_y,
        d_z,
        d_nodelist,
        d_volo,
        d_delv_eta,
        d_delx_eta,
        d_delv_zeta,
        d_delx_zeta,
        d_delv_xi,
        d_delx_xi,
        d_vnew,
        numElem );

    //=========================================================
    // CalcMonotonicQForElems(domain, vnew) ;
    //=========================================================
    Real_t monoq_limiter_mult = domain.monoq_limiter_mult();
    Real_t monoq_max_slope = domain.monoq_max_slope();
    Real_t qlc_monoq = domain.qlc_monoq();
    Real_t qqc_monoq = domain.qqc_monoq();

    launch_calcMonotonicQForElems(gws_elem, lws, 
        d_elemBC,
        d_elemMass,
        d_ql,
        d_qq,
        d_vdov,
        d_volo,
        d_delv_eta,
        d_delx_eta,
        d_delv_zeta,
        d_delx_zeta,
        d_delv_xi,
        d_delx_xi,
        d_lxim,
        d_lxip,
        d_lzetam,
        d_lzetap,
        d_letap,
        d_letam,
        d_vnew,
        monoq_limiter_mult,
        monoq_max_slope,
        qlc_monoq,
        qqc_monoq,
        numElem );

#ifdef DEBUG
    Real_t* qq_tmp = (Real_t*) malloc (sizeof(Real_t)*numElem);
    Real_t* ql_tmp = (Real_t*) malloc (sizeof(Real_t)*numElem);

    d_qq->read(qq_tmp,sizeof(Real_t)*numElem);
    d_ql->read(ql_tmp,sizeof(Real_t)*numElem);
    for (int i = 0; i < numElem; i++) {
      printf("mqr: %d %f %f\n", i, qq_tmp[i], ql_tmp[i]);
    }

    free(qq_tmp);
    free(ql_tmp);
#endif

#ifdef VERIFY
    /* Don't allow excessive artificial viscosity */
    Index_t idx = -1;
    for (Index_t i=0; i<numElem; ++i) {
      if ( domain.q(i) > domain.qstop() ) {
        idx = i ;
        break ;
      }
    }

    if(idx >= 0) {
      printf("QStopError\n");
      //exit(QStopError);
    }
#endif

    //=================================================
    //ApplyMaterialPropertiesForElems(domain, vnew) ;
    //=================================================
    Real_t  e_cut = domain.e_cut() ;
    Real_t  p_cut = domain.p_cut() ;
    //Real_t  ss4o3 = domain.ss4o3() ;
    Real_t  q_cut = domain.q_cut() ;
    Real_t  v_cut = domain.v_cut() ;

    Real_t eosvmax = domain.eosvmax() ;
    Real_t eosvmin = domain.eosvmin() ;
    Real_t pmin    = domain.pmin() ;
    Real_t emin    = domain.emin() ;
    Real_t rho0    = domain.refdens() ;

    launch_applyMaterialPropertiesForElems(gws_elem, lws, 
        d_ql,
        d_qq,
        d_delv,
        d_elemRep,
        d_elemElem,
        d_q,
        d_p,
        d_e,
        d_ss,
        d_v,
        d_vnew,
        e_cut,
        p_cut,
        //ss4o3,
        q_cut,
        v_cut,

        eosvmax,
        eosvmin,
        pmin,
        emin,
        rho0,
        numElem );

    d_arealg->read(arealg,sizeof(Real_t)*numElem);
    d_ss->read(ss,sizeof(Real_t)*numElem);

#ifdef DEBUG
    d_p->read(p,sizeof(Real_t)*numElem);
    d_q->read(q,sizeof(Real_t)*numElem);
    d_e->read(e,sizeof(Real_t)*numElem);
    d_v->read(v,sizeof(Real_t)*numElem);
    for (int i = 0; i < numElem; i++) {
      printf("eos: %f %f %f %f %f\n", q[i], p[i], e[i], ss[i], v[i]);
    }
#endif

    // TODO offload to a device
    CalcTimeConstraintsForElems(domain);

    if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
      printf("cycle = %d, time = %e, dt=%e\n",
          locDom->cycle(), double(locDom->time()), double(locDom->deltatime()) ) ;
    }
    if (opts.iteration_cap == 1){
      break;
    }
    opts.iteration_cap -= 1;
  } // while

  d_e->read(e,sizeof(Real_t)*numElem);

  // Use reduced max elapsed time
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double elapsed_timeG = time * 1e-9;

  // Write out final viz file */
  if (opts.viz) {
    DumpToVisit(*locDom, opts.numFiles, myRank, numRanks) ;
  }

  if ((myRank == 0) && (opts.quiet == 0)) {
    VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, opts.nx, numRanks);
  }

  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *out=fopen(path,"wb");if(!out)exit(5);
    int meta[]={numNode,numElem,locDom->cycle()};fwrite(meta,sizeof(int),3,out);
    Real_t timeinfo[]={locDom->time(),locDom->deltatime(),locDom->dtcourant(),locDom->dthydro()};
    fwrite(timeinfo,sizeof(Real_t),4,out);
    const size_t maxsize=numNode>numElem?numNode:numElem;
    std::vector<Real_t> snapshot(maxsize);
    for(auto entry: std::vector<std::pair<Buffer *,size_t>>{
      {d_x,static_cast<size_t>(numNode)},
      {d_y,static_cast<size_t>(numNode)},
      {d_z,static_cast<size_t>(numNode)},
      {d_xd,static_cast<size_t>(numNode)},
      {d_yd,static_cast<size_t>(numNode)},
      {d_zd,static_cast<size_t>(numNode)},
      {d_xdd,static_cast<size_t>(numNode)},
      {d_ydd,static_cast<size_t>(numNode)},
      {d_zdd,static_cast<size_t>(numNode)},
      {d_fx,static_cast<size_t>(numNode)},
      {d_fy,static_cast<size_t>(numNode)},
      {d_fz,static_cast<size_t>(numNode)},
      {d_e,static_cast<size_t>(numElem)},
      {d_p,static_cast<size_t>(numElem)},
      {d_q,static_cast<size_t>(numElem)},
      {d_v,static_cast<size_t>(numElem)},
      {d_ss,static_cast<size_t>(numElem)},
      {d_vdov,static_cast<size_t>(numElem)},
      {d_delv,static_cast<size_t>(numElem)},
      {d_arealg,static_cast<size_t>(numElem)},
      {d_ql,static_cast<size_t>(numElem)},
      {d_qq,static_cast<size_t>(numElem)}
    }) {entry.first->read(snapshot.data(),entry.second*sizeof(Real_t));fwrite(snapshot.data(),sizeof(Real_t),entry.second,out);}
    fclose(out);
  }
  //  Release resources
  delete d_x;
  delete d_y;
  delete d_z;
  delete d_fx;
  delete d_fy;
  delete d_fz;
  delete d_xd;
  delete d_yd;
  delete d_zd;
  delete d_xdd;
  delete d_ydd;
  delete d_zdd;
  delete d_nodelist;
  delete d_determ;
  delete d_fx_elem;
  delete d_fy_elem;
  delete d_fz_elem;
  delete d_dvdx;
  delete d_dvdy;
  delete d_dvdz;
  delete d_x8n;
  delete d_y8n;
  delete d_z8n;
  delete d_sigxx;
  delete d_sigyy;
  delete d_sigzz;
  delete d_delv_xi;
  delete d_delx_xi;
  delete d_delv_eta;
  delete d_delx_eta;
  delete d_delv_zeta;
  delete d_delx_zeta;
  delete d_p;
  delete d_q;
  delete d_volo;
  delete d_v;
  delete d_vol_error;
  delete d_nodeElemStart;
  delete d_nodeElemCornerList;
  delete d_gamma;
  delete d_ss;
  delete d_elemMass;
  delete d_nodalMass;
  delete d_symmX;
  delete d_symmY;
  delete d_symmZ;
  delete d_vdov;
  delete d_delv;
  delete d_arealg;
  delete d_dxx;
  delete d_dyy;
  delete d_dzz;
  delete d_vnew;
  delete d_lzetam;
  delete d_lzetap;
  delete d_letap;
  delete d_letam;
  delete d_lxip;
  delete d_lxim;
  delete d_elemBC;
  delete d_ql;
  delete d_qq;
  delete d_e;
  delete d_elemRep;
  delete d_elemElem;
  locDom->DeallocateGradients();
  locDom->DeallocateStrains();
  Release(&determ);
  delete(locDom);

  platform.finalize();
  return 0 ;
}
