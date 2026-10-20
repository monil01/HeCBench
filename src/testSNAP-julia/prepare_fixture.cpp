#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <fstream>
#include <cstdint>
#define __device__
#include "original_native/snap.h"
#include "original_native/utils.cu"
#if REFDATA_TWOJ == 14
#include "refdata_2J14_W.h"
#elif REFDATA_TWOJ == 4
#include "refdata_2J4_W.h"
#else
#include "refdata_2J2_W.h"
#endif
int nsteps=1;
template<class T>void put(std::ofstream&o,const T*p,size_t n){o.write(reinterpret_cast<const char*>(p),sizeof(T)*n);if(!o)throw std::runtime_error("fixture write");}
#define malloc(n) calloc(1,n)
int main(int argc,char**argv){if(argc!=2)return 2;
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


int jt=0,jjt=0;
std::vector<int32_t>neighbors(num_atoms*num_nbor);
for(int natom=0;natom<num_atoms;natom++)for(int nbor=0;nbor<num_nbor;nbor++){
 rij[ULIST_INDEX(natom,nbor,0)]=refdata.rij[jt++];rij[ULIST_INDEX(natom,nbor,1)]=refdata.rij[jt++];rij[ULIST_INDEX(natom,nbor,2)]=refdata.rij[jt++];neighbors[INDEX_2D(natom,nbor)]=refdata.jlist[jjt++];wj[INDEX_2D(natom,nbor)]=1.;rcutij[INDEX_2D(natom,nbor)]=rcutfac;
}
std::vector<double>reference(ntotal*3);for(int i=0;i<ntotal;i++)for(int k=0;k<3;k++)reference[i+ntotal*k]=refdata.fj[3*i+k];
uint32_t h[]={0x534e4150u,uint32_t(twojmax),uint32_t(num_atoms),uint32_t(nghost),uint32_t(num_nbor),uint32_t(ncoeff),uint32_t(idxu_max),uint32_t(idxdu_max),uint32_t(idxz_max),uint32_t(idxcg_max),uint32_t(jdim),uint32_t(jdimpq)};
double scalars[]={rcutfac,wself,rfac0,rmin0};std::ofstream out(argv[1],std::ios::binary);put(out,h,12);put(out,scalars,4);
put(out,idxu_block,jdim);put(out,ulist_parity,idxu_max);put(out,idxdu_block,jdim);put(out,idxz,idxz_max*9);put(out,idxcg_block,jdim*jdim*jdim);put(out,rootpqarray,jdimpq*jdimpq);put(out,cglist,idxcg_max);put(out,idxzbeta,idxz_max);put(out,rij,num_atoms*num_nbor*3);put(out,neighbors.data(),neighbors.size());put(out,wj,num_atoms*num_nbor);put(out,rcutij,num_atoms*num_nbor);put(out,reference.data(),reference.size());put(out,coeffi,ncoeff+1);
printf("SNAP_FIXTURE J=%d atoms=%d ghost=%d nbor=%d idxu=%d idxdu=%d idxz=%d idxcg=%d\n",twojmax,num_atoms,nghost,num_nbor,idxu_max,idxdu_max,idxz_max,idxcg_max);return 0;}
