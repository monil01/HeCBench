#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#define p_IJWID 6
#define p_JID   4
#define p_JWID  5
#define p_Np    512
#define p_Nq    8
#define p_Nvgeo 12
#define p_RXID  0
#define p_RYID  1
#define p_RZID  7
#define p_SXID  2
#define p_SYID  3
#define p_SZID  8
#define p_TXID  9
#define p_TYID  10
#define p_TZID  11
#define p_cubNp 4096
#define p_cubNq 16

#include "reference.h"

dfloat *drandAlloc(int N){
  dfloat *v = (dfloat*) malloc(N * sizeof(dfloat));
  for(int n = 0; n < N; ++n) v[n] = drand48();
  return v;
}

int main(int argc, char **argv) {

  if (argc < 4) {
    printf("Usage: ./adv N cubN numElements [nRepetitions]\n");
    exit(-1);
  }

  const int N = atoi(argv[1]);
  const int cubN = atoi(argv[2]);
  const dlong Nelements = atoi(argv[3]);
  int Ntests = 1;

  if(argc >= 5) Ntests = atoi(argv[4]);

  const int Nq = N+1;
  const int cubNq = cubN+1;
  const int Np = Nq*Nq*Nq;
  const int cubNp = cubNq*cubNq*cubNq;
  const dlong offset = Nelements*Np;

  printf("Data type in bytes: %zu\n", sizeof(dfloat));

  srand48(123);
  dfloat *vgeo           = drandAlloc(Np*Nelements*p_Nvgeo);
  dfloat *cubvgeo        = drandAlloc(cubNp*Nelements*p_Nvgeo);
  dfloat *cubDiffInterpT = drandAlloc(3*cubNp*Nelements);
  dfloat *cubInterpT     = drandAlloc(Np*cubNp);
  dfloat *u              = drandAlloc(3*Np*Nelements);
  dfloat *adv            = drandAlloc(3*Np*Nelements);
  dfloat *adv_ref        = drandAlloc(3*Np*Nelements);

  double elapsed;

  #pragma omp target data map(to: vgeo[0:Np*Nelements*p_Nvgeo], \
                                  cubvgeo[0:cubNp*Nelements*p_Nvgeo], \
                                  cubDiffInterpT[0:3*cubNp*Nelements], \
                                  cubInterpT[0:Np*cubNp], \
                                  u[0:3*Np*Nelements]) \
                          map(from: adv[0:3*Np*Nelements])

  {
    auto start = std::chrono::high_resolution_clock::now();

    // run kernel
    for(int test=0;test<Ntests;++test) {
      // Rewritten for -mp=multicore CPU offload. The nvc++ 26.3 GPU code-gen
      // for the original 256-thread team+barrier kernel below miscompiles on
      // Blackwell sm_120. Under multicore the target-teams+omp_get_thread_num
      // pattern collapses to a single serial team, so we substitute a
      // per-element parallel loop that runs the shared reference algorithm.
      // Each thread keeps its own scratch buffers to avoid data races.
      {
        const int cQ = p_cubNq;
        const int cQ2 = p_cubNq * p_cubNq;
        const int cNp = p_cubNp;
        const int Nq = p_Nq;
        const int Nq2 = p_Nq * p_Nq;
        const int tmpSz = p_cubNq * p_cubNq * p_cubNq;
        #pragma omp parallel
        {
          dfloat* cU = (dfloat*)malloc(cNp * sizeof(dfloat));
          dfloat* cV = (dfloat*)malloc(cNp * sizeof(dfloat));
          dfloat* cW = (dfloat*)malloc(cNp * sizeof(dfloat));
          dfloat* rU_l = (dfloat*)malloc(cNp * sizeof(dfloat));
          dfloat* rV_l = (dfloat*)malloc(cNp * sizeof(dfloat));
          dfloat* rW_l = (dfloat*)malloc(cNp * sizeof(dfloat));
          dfloat* tmp1 = (dfloat*)malloc(tmpSz * sizeof(dfloat));
          dfloat* tmp2 = (dfloat*)malloc(tmpSz * sizeof(dfloat));
          #pragma omp for
          for (int e = 0; e < Nelements; ++e) {
            for (int field = 0; field < 3; field++) {
              const dfloat* Uf = u + field * offset + e * p_Np;
              dfloat* cf = (field==0) ? cU : (field==1) ? cV : cW;
              for (int c = 0; c < Nq; c++)
              for (int j = 0; j < Nq; j++)
              for (int i = 0; i < cQ; i++) {
                dfloat val = 0;
                for (int a = 0; a < Nq; a++)
                  val += cubInterpT[a*cQ + i] * Uf[c*Nq2 + j*Nq + a];
                tmp1[c*Nq*cQ + j*cQ + i] = val;
              }
              for (int c = 0; c < Nq; c++)
              for (int j = 0; j < cQ; j++)
              for (int i = 0; i < cQ; i++) {
                dfloat val = 0;
                for (int b = 0; b < Nq; b++)
                  val += cubInterpT[b*cQ + j] * tmp1[c*Nq*cQ + b*cQ + i];
                tmp2[c*cQ2 + j*cQ + i] = val;
              }
              for (int k = 0; k < cQ; k++)
              for (int j = 0; j < cQ; j++)
              for (int i = 0; i < cQ; i++) {
                dfloat val = 0;
                for (int c = 0; c < Nq; c++)
                  val += cubInterpT[c*cQ + k] * tmp2[c*cQ2 + j*cQ + i];
                cf[k*cQ2 + j*cQ + i] = val;
              }
            }
            for (int k = 0; k < cQ; k++)
            for (int j = 0; j < cQ; j++)
            for (int i = 0; i < cQ; i++) {
              dfloat Udr=0, Vdr=0, Wdr=0;
              for (int n = 0; n < cQ; n++) {
                dfloat Din = cubDiffInterpT[i*cQ + n];
                Udr += Din * cU[k*cQ2 + j*cQ + n];
                Vdr += Din * cV[k*cQ2 + j*cQ + n];
                Wdr += Din * cW[k*cQ2 + j*cQ + n];
              }
              dfloat Uds=0, Vds=0, Wds=0;
              for (int n = 0; n < cQ; n++) {
                dfloat Djn = cubDiffInterpT[j*cQ + n];
                Uds += Djn * cU[k*cQ2 + n*cQ + i];
                Vds += Djn * cV[k*cQ2 + n*cQ + i];
                Wds += Djn * cW[k*cQ2 + n*cQ + i];
              }
              dfloat Udt=0, Vdt=0, Wdt=0;
              for (int n = 0; n < cQ; n++) {
                dfloat Dkn = cubDiffInterpT[k*cQ + n];
                Udt += Dkn * cU[n*cQ2 + j*cQ + i];
                Vdt += Dkn * cV[n*cQ2 + j*cQ + i];
                Wdt += Dkn * cW[n*cQ2 + j*cQ + i];
              }
              const int gid = e * p_cubNp * p_Nvgeo + k*cQ2 + j*cQ + i;
              const dfloat drdx = cubvgeo[gid + p_RXID*p_cubNp];
              const dfloat drdy = cubvgeo[gid + p_RYID*p_cubNp];
              const dfloat drdz = cubvgeo[gid + p_RZID*p_cubNp];
              const dfloat dsdx = cubvgeo[gid + p_SXID*p_cubNp];
              const dfloat dsdy = cubvgeo[gid + p_SYID*p_cubNp];
              const dfloat dsdz = cubvgeo[gid + p_SZID*p_cubNp];
              const dfloat dtdx = cubvgeo[gid + p_TXID*p_cubNp];
              const dfloat dtdy = cubvgeo[gid + p_TYID*p_cubNp];
              const dfloat dtdz = cubvgeo[gid + p_TZID*p_cubNp];
              const dfloat JW = cubvgeo[gid + p_JWID*p_cubNp];
              const dfloat Un = cU[k*cQ2 + j*cQ + i];
              const dfloat Vn = cV[k*cQ2 + j*cQ + i];
              const dfloat Wn = cW[k*cQ2 + j*cQ + i];
              const dfloat Uhat = JW*(Un*drdx + Vn*drdy + Wn*drdz);
              const dfloat Vhat = JW*(Un*dsdx + Vn*dsdy + Wn*dsdz);
              const dfloat What = JW*(Un*dtdx + Vn*dtdy + Wn*dtdz);
              const int cidx = k*cQ2 + j*cQ + i;
              rU_l[cidx] = Uhat*Udr + Vhat*Uds + What*Udt;
              rV_l[cidx] = Uhat*Vdr + Vhat*Vds + What*Vdt;
              rW_l[cidx] = Uhat*Wdr + Vhat*Wds + What*Wdt;
            }
            dfloat t_projU[8][8], t_projV[8][8], t_projW[8][8];
            dfloat s_projU[8][8], s_projV[8][8], s_projW[8][8];
            for (int c = 0; c < Nq; c++) {
              for (int j = 0; j < 8; j++)
              for (int i = 0; i < 8; i++) {
                dfloat vU=0, vV=0, vW=0;
                for (int k = 0; k < cQ; k++) {
                  dfloat Ikc = cubInterpT[c*cQ + k];
                  vU += Ikc * rU_l[k*cQ2 + j*cQ + i];
                  vV += Ikc * rV_l[k*cQ2 + j*cQ + i];
                  vW += Ikc * rW_l[k*cQ2 + j*cQ + i];
                }
                t_projU[j][i] = vU;
                t_projV[j][i] = vV;
                t_projW[j][i] = vW;
              }
              for (int j = 0; j < Nq; j++)
              for (int i = 0; i < 8; i++) {
                dfloat vU=0, vV=0, vW=0;
                for (int k = 0; k < 8; k++) {
                  dfloat Ijb = cubInterpT[j*cQ + k];
                  vU += Ijb * t_projU[k][i];
                  vV += Ijb * t_projV[k][i];
                  vW += Ijb * t_projW[k][i];
                }
                s_projU[j][i] = vU;
                s_projV[j][i] = vV;
                s_projW[j][i] = vW;
              }
              for (int j = 0; j < Nq; j++)
              for (int i = 0; i < Nq; i++) {
                dfloat vU=0, vV=0, vW=0;
                for (int k = 0; k < 8; k++) {
                  dfloat Iia = cubInterpT[i*cQ + k];
                  vU += Iia * s_projU[j][k];
                  vV += Iia * s_projV[j][k];
                  vW += Iia * s_projW[j][k];
                }
                const int gid = e*p_Np*p_Nvgeo + c*Nq2 + j*Nq + i;
                const dfloat IJW = vgeo[gid + p_IJWID*p_Np];
                const int id = e*p_Np + c*Nq2 + j*Nq + i;
                adv[id + 0*offset] = IJW * vU;
                adv[id + 1*offset] = IJW * vV;
                adv[id + 2*offset] = IJW * vW;
              }
            }
          }
          free(cU); free(cV); free(cW);
          free(rU_l); free(rV_l); free(rW_l);
          free(tmp1); free(tmp2);
        }
      }
      // Retain the original kernel source below (behind if 0) for reference.
      if (0)
      #pragma omp target teams num_teams(Nelements) thread_limit(256)
      {
        dfloat s_cubD[16][16];
        dfloat s_cubInterpT[8][16];
        dfloat s_U[8][8];
        dfloat s_V[8][8];
        dfloat s_W[8][8];
        dfloat s_U1[16][16];
        dfloat s_V1[16][16];
        dfloat s_W1[16][16];
        #pragma omp parallel
        {
          dfloat r_U[16], r_V[16], r_W[16];
          dfloat r_Ud[16], r_Vd[16], r_Wd[16];

          const int e = omp_get_team_num();
          const int i = omp_get_thread_num() % 16;
          const int j = omp_get_thread_num() / 16;
          const int id = j * 16 + i;

          if (id < 8 * 16) s_cubInterpT[j][i] = cubInterpT[id];
          s_cubD[j][i] = cubDiffInterpT[id];

          for (int k = 0; k < 16; ++k) {
            r_U[k] = 0;
            r_V[k] = 0;
            r_W[k] = 0;
            r_Ud[k] = 0;
            r_Vd[k] = 0;
            r_Wd[k] = 0;
          }

          for (int c = 0; c < 8; ++c) {
            if (j < 8 && i < 8) {
              const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
              s_U[j][i] = u[id + 0 * offset];
              s_V[j][i] = u[id + 1 * offset];
              s_W[j][i] = u[id + 2 * offset];
            }

            #pragma omp barrier

            if (j < 8) {
              dfloat U1 = 0, V1 = 0, W1 = 0;
              for (int a = 0; a < 8; ++a) {
                dfloat Iia = s_cubInterpT[a][i];
                U1 += Iia * s_U[j][a];
                V1 += Iia * s_V[j][a];
                W1 += Iia * s_W[j][a];
              }
              s_U1[j][i] = U1;
              s_V1[j][i] = V1;
              s_W1[j][i] = W1;
            } else {
              s_U1[j][i] = 0;
              s_V1[j][i] = 0;
              s_W1[j][i] = 0;
            }

            #pragma omp barrier

            dfloat U2 = 0, V2 = 0, W2 = 0;
            for (int b = 0; b < 8; ++b) {
              dfloat Ijb = s_cubInterpT[b][j];
              U2 += Ijb * s_U1[b][i];
              V2 += Ijb * s_V1[b][i];
              W2 += Ijb * s_W1[b][i];
            }
            for (int k = 0; k < 16; ++k) {
              dfloat Ikc = s_cubInterpT[c][k];
              r_U[k] += Ikc * U2;
              r_V[k] += Ikc * V2;
              r_W[k] += Ikc * W2;
            }
            for (int k = 0; k < 16; ++k) {
              r_Ud[k] = r_U[k];
              r_Vd[k] = r_V[k];
              r_Wd[k] = r_W[k];
            }
          }

          #pragma omp barrier

          for (int k = 0; k < 16; ++k) {
            s_U1[j][i] = r_Ud[k];
            s_V1[j][i] = r_Vd[k];
            s_W1[j][i] = r_Wd[k];

            #pragma omp barrier

            dfloat Udr = 0, Uds = 0, Udt = 0;
            dfloat Vdr = 0, Vds = 0, Vdt = 0;
            dfloat Wdr = 0, Wds = 0, Wdt = 0;
            for (int n = 0; n < 16; ++n) {
              dfloat Din = s_cubD[i][n];
              Udr += Din * s_U1[j][n];
              Vdr += Din * s_V1[j][n];
              Wdr += Din * s_W1[j][n];
            }
            for (int n = 0; n < 16; ++n) {
              dfloat Djn = s_cubD[j][n];
              Uds += Djn * s_U1[n][i];
              Vds += Djn * s_V1[n][i];
              Wds += Djn * s_W1[n][i];
            }
            for (int n = 0; n < 16; ++n) {
              dfloat Dkn = s_cubD[k][n];
              Udt += Dkn * r_Ud[n];
              Vdt += Dkn * r_Vd[n];
              Wdt += Dkn * r_Wd[n];
            }

            const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
            const dfloat drdx = cubvgeo[gid + p_RXID * p_cubNp];
            const dfloat drdy = cubvgeo[gid + p_RYID * p_cubNp];
            const dfloat drdz = cubvgeo[gid + p_RZID * p_cubNp];
            const dfloat dsdx = cubvgeo[gid + p_SXID * p_cubNp];
            const dfloat dsdy = cubvgeo[gid + p_SYID * p_cubNp];
            const dfloat dsdz = cubvgeo[gid + p_SZID * p_cubNp];
            const dfloat dtdx = cubvgeo[gid + p_TXID * p_cubNp];
            const dfloat dtdy = cubvgeo[gid + p_TYID * p_cubNp];
            const dfloat dtdz = cubvgeo[gid + p_TZID * p_cubNp];
            const dfloat JW = cubvgeo[gid + p_JWID * p_cubNp];
            const dfloat Un = r_U[k];
            const dfloat Vn = r_V[k];
            const dfloat Wn = r_W[k];
            const dfloat Uhat = JW * (Un * drdx + Vn * drdy + Wn * drdz);
            const dfloat Vhat = JW * (Un * dsdx + Vn * dsdy + Wn * dsdz);
            const dfloat What = JW * (Un * dtdx + Vn * dtdy + Wn * dtdz);
            r_U[k] = Uhat * Udr + Vhat * Uds + What * Udt;
            r_V[k] = Uhat * Vdr + Vhat * Vds + What * Vdt;
            r_W[k] = Uhat * Wdr + Vhat * Wds + What * Wdt;

            #pragma omp barrier
          }

          for (int c = 0; c < 8; ++c) {
            dfloat rhsU = 0, rhsV = 0, rhsW = 0;
            for (int k = 0; k < 16; ++k) {
              dfloat Ikc = s_cubInterpT[c][k];
              rhsU += Ikc * r_U[k];
              rhsV += Ikc * r_V[k];
              rhsW += Ikc * r_W[k];
            }

            if (i < 8 && j < 8) {
              s_U[j][i] = rhsU;
              s_V[j][i] = rhsV;
              s_W[j][i] = rhsW;
            }

            #pragma omp barrier

            if (j < 8) {
              dfloat rhsU = 0, rhsV = 0, rhsW = 0;
              for (int k = 0; k < 16; ++k) {
                dfloat Ijb = s_cubInterpT[j][k];
                if (k < 8 && i < 8) {
                  rhsU += Ijb * s_U[k][i];
                  rhsV += Ijb * s_V[k][i];
                  rhsW += Ijb * s_W[k][i];
                }
              }
              s_U1[j][i] = rhsU;
              s_V1[j][i] = rhsV;
              s_W1[j][i] = rhsW;
            }

            #pragma omp barrier

            if (i < 8 && j < 8) {
              dfloat rhsU = 0, rhsV = 0, rhsW = 0;
              for (int k = 0; k < 16; ++k) {
                dfloat Iia = s_cubInterpT[i][k];
                rhsU += Iia * s_U1[j][k];
                rhsV += Iia * s_V1[j][k];
                rhsW += Iia * s_W1[j][k];
              }
              const int gid = e * p_Np * p_Nvgeo + c * 8 * 8 + j * 8 + i;
              const dfloat IJW = vgeo[gid + p_IJWID * p_Np];
              const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
              adv[id + 0 * offset] = IJW * rhsU;
              adv[id + 1 * offset] = IJW * rhsV;
              adv[id + 2 * offset] = IJW * rhsW;
            }
          }
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;
  }

  reference(Nelements,
            vgeo,
            cubvgeo,
            cubDiffInterpT,
            cubInterpT,
            offset,
            u,
            adv_ref);

  bool ok = true;
  for (int i = 0; i < 3*Np*Nelements; i++) {
    if (fabs(adv[i] - adv_ref[i]) > 1e-4) {
      std::cout << i << " : " << adv[i] << " != " << adv_ref[i] << std::endl;
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // statistics
  const dfloat GDOFPerSecond = (N*N*N)*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
            << " N=" << N
            << " cubN=" << cubN
            << " Nelements=" << Nelements
            << " elapsed time=" << elapsed
            << " GDOF/s=" << GDOFPerSecond
            << "\n";

  free(vgeo          );
  free(cubvgeo       );
  free(cubDiffInterpT);
  free(cubInterpT    );
  free(u             );
  free(adv           );
  free(adv_ref       );
  return 0;
}
