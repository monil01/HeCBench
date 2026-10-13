#include <chrono>
#include <math.h>
#include <stdio.h>
#include "dmem.h"

void* cp_to_device(char* from,size_t size) {
  void* p; allocDeviceBuffer(&p,size); copyToDevice(p,from,size); return p;
}
void cp_to_host(char* to,void* from,size_t size) {copyFromDevice(to,from,size);freeDeviceBuffer(from);}

      struct full_data
      {
        int sizex;
        int sizey;
        int Nmats;
        double * __restrict__ rho;
        double * __restrict__ rho_mat_ave;
        double * __restrict__ p;
        double * __restrict__ Vf;
        double * __restrict__ t;
        double * __restrict__ V;
        double * __restrict__ x;
        double * __restrict__ y;
        double * __restrict__ n;
        double * __restrict__ rho_ave;
      };

      struct compact_data
      {
        int sizex;
        int sizey;
        int Nmats;
        double * __restrict__ rho_compact;
        double * __restrict__ rho_compact_list;
        double * __restrict__ rho_mat_ave_compact;
        double * __restrict__ rho_mat_ave_compact_list;
        double * __restrict__ p_compact;
        double * __restrict__ p_compact_list;
        double * __restrict__ Vf_compact_list;
        double * __restrict__ t_compact;
        double * __restrict__ t_compact_list;
        double * __restrict__ V;
        double * __restrict__ x;
        double * __restrict__ y;
        double * __restrict__ n;
        double * __restrict__ rho_ave_compact;
        int * __restrict__ imaterial;
        int * __restrict__ matids;
        int * __restrict__ nextfrac;
        int * __restrict__ mmc_index;
        int * __restrict__ mmc_i;
        int * __restrict__ mmc_j;
        int mm_len;
        int mmc_cells;
      };

      void compact_cell_centric(full_data cc, compact_data ccc, int argc, char** argv)
      {
        int sizex = cc.sizex;
        int sizey = cc.sizey;
        int Nmats = cc.Nmats;
        int mmc_cells = ccc.mmc_cells;
        int mm_len = ccc.mm_len;

        void* d_imaterial = cp_to_device((char*)ccc.imaterial, sizex*sizey*sizeof(int));
        void* d_matids = cp_to_device((char*)ccc.matids, mm_len*sizeof(int));
        void* d_nextfrac = cp_to_device((char*)ccc.nextfrac, mm_len*sizeof(int));
        void* d_mmc_index = cp_to_device((char*)ccc.mmc_index, (mmc_cells+1)*sizeof(int));
        void* d_mmc_i = cp_to_device((char*)ccc.mmc_i, (mmc_cells)*sizeof(int));
        void* d_mmc_j = cp_to_device((char*)ccc.mmc_j, (mmc_cells)*sizeof(int));
        void* d_x = cp_to_device((char*)ccc.x, sizex*sizey*sizeof(double));
        void* d_y = cp_to_device((char*)ccc.y, sizex*sizey*sizeof(double));
        void* d_rho_compact = cp_to_device((char*)ccc.rho_compact, sizex*sizey*sizeof(double));
        void* d_rho_compact_list = cp_to_device((char*)ccc.rho_compact_list,mm_len*sizeof(double));
        void* d_rho_mat_ave_compact = cp_to_device((char*)ccc.rho_mat_ave_compact, sizex*sizey*sizeof(double));
        void* d_rho_mat_ave_compact_list = cp_to_device((char*)ccc.rho_mat_ave_compact_list,mm_len*sizeof(double));
        void* d_p_compact = cp_to_device((char*)ccc.p_compact, sizex*sizey*sizeof(double));
        void* d_p_compact_list = cp_to_device((char*)ccc.p_compact_list,mm_len*sizeof(double));
        void* d_t_compact = cp_to_device((char*)ccc.t_compact, sizex*sizey*sizeof(double));
        void* d_t_compact_list = cp_to_device((char*)ccc.t_compact_list,mm_len*sizeof(double));
        void* d_Vf_compact_list = cp_to_device((char*)ccc.Vf_compact_list, mm_len*sizeof(double));
        void* d_V = cp_to_device((char*)ccc.V, sizex*sizey*sizeof(double));
        void* d_n = cp_to_device((char*)ccc.n, Nmats*sizeof(double));
        void* d_rho_ave_compact = cp_to_device((char*)ccc.rho_ave_compact, sizex*sizey*sizeof(double));

        int thx = 32;
        int thy = 4;
        dim3 threads(thx,thy,1);
        dim3 blocks((sizex-1)/thx+1, (sizey-1)/thy+1, 1);

        // Cell-centric algorithms
        // Computational loop 1 - average density in cell
        
        auto t0 = std::chrono::system_clock::now();
        launchMulti("ccc_loop1", dim3(blocks), dim3(threads), {memory(d_imaterial), memory(d_nextfrac), memory(d_rho_compact), memory(d_rho_compact_list), memory(d_Vf_compact_list), memory(d_V), memory(d_rho_ave_compact), scalar(sizex), scalar(sizey), memory(d_mmc_index)});
#ifndef FUSED
        launchMulti("ccc_loop1_2", dim3((mmc_cells-1)/(thx*thy)+1), dim3((thx*thy)), {memory(d_rho_compact_list), memory(d_Vf_compact_list), memory(d_V), memory(d_rho_ave_compact), memory(d_mmc_index), scalar(mmc_cells), memory(d_mmc_i), memory(d_mmc_j), scalar(sizex), scalar(sizey)});
#endif
        
        std::chrono::duration<double> t1 = std::chrono::system_clock::now() - t0;
        printf("Compact matrix, cell centric, alg 1: %g msec\n", t1.count() * 1000);

        // Computational loop 2 - Pressure for each cell and each material
        t0 = std::chrono::system_clock::now();
        launchMulti("ccc_loop2", dim3(blocks), dim3(threads), {memory(d_imaterial), memory(d_matids), memory(d_nextfrac), memory(d_rho_compact), memory(d_rho_compact_list), memory(d_t_compact), memory(d_t_compact_list), memory(d_Vf_compact_list), memory(d_n), memory(d_p_compact), memory(d_p_compact_list), scalar(sizex), scalar(sizey), memory(d_mmc_index)});
#ifndef FUSED
        launchMulti("ccc_loop2_2", dim3((mm_len-1)/(thx*thy)+1), dim3((thx*thy)), {memory(d_matids), memory(d_rho_compact_list), memory(d_t_compact_list), memory(d_Vf_compact_list), memory(d_n), memory(d_p_compact_list), memory(d_mmc_index), scalar(mm_len)});
#endif
        
        std::chrono::duration<double> t2 = std::chrono::system_clock::now() - t0;
        printf("Compact matrix, cell centric, alg 2: %g msec\n", t2.count() * 1000);

        // Computational loop 3 - Average density of each material over neighborhood of each cell
        t0 = std::chrono::system_clock::now();
        launchMulti("ccc_loop3", dim3(blocks), dim3(threads), {memory(d_imaterial), memory(d_nextfrac), memory(d_matids), memory(d_rho_compact), memory(d_rho_compact_list), memory(d_rho_mat_ave_compact), memory(d_rho_mat_ave_compact_list), memory(d_x), memory(d_y), scalar(sizex), scalar(sizey), memory(d_mmc_index)});  
        
        std::chrono::duration<double> t3 = std::chrono::system_clock::now() - t0;
        printf("Compact matrix, cell centric, alg 3: %g msec\n", t3.count() * 1000);

        cp_to_host((char*)ccc.x, d_x, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.y, d_y, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.rho_compact, d_rho_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.rho_compact_list, d_rho_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.rho_mat_ave_compact, d_rho_mat_ave_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.rho_mat_ave_compact_list, d_rho_mat_ave_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.p_compact, d_p_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.p_compact_list, d_p_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.t_compact, d_t_compact, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.t_compact_list, d_t_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.Vf_compact_list, d_Vf_compact_list, mm_len*sizeof(double));
        cp_to_host((char*)ccc.V, d_V, sizex*sizey*sizeof(double));
        cp_to_host((char*)ccc.n, d_n, Nmats*sizeof(double));
        cp_to_host((char*)ccc.rho_ave_compact, d_rho_ave_compact, sizex*sizey*sizeof(double));

        // cudaFree are included in cp_to_host
      }

      bool compact_check_results(full_data cc, compact_data ccc)
      {
        int sizex = cc.sizex;
        int sizey = cc.sizey;
        int Nmats = cc.Nmats;
        //int mmc_cells = ccc.mmc_cells;
        //int mm_len = ccc.mm_len;

        printf("Checking results of compact representation... ");

        for (int j = 0; j < sizey; j++) {
          for (int i = 0; i < sizex; i++) {
            if (fabs(cc.rho_ave[i+sizex*j] - ccc.rho_ave_compact[i+sizex*j]) > 0.0001) {
              printf("1. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d)\n",
                  cc.rho_ave[i+sizex*j], ccc.rho_ave_compact[i+sizex*j], i, j);
              return false;
            }
            int ix = ccc.imaterial[i+sizex*j];
            if (ix <= 0) {
#ifdef LINKED
              for (ix = -ix; ix >= 0; ix = ccc.nextfrac[ix]) {
#else
              for (int ix = ccc.mmc_index[-ccc.imaterial[i+sizex*j]]; ix < ccc.mmc_index[-ccc.imaterial[i+sizex*j]+1]; ix++) {
#endif
                int mat = ccc.matids[ix];
                if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - ccc.p_compact_list[ix]) > 0.0001) {
                  printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                      cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact_list[ix], i, j, mat);
                  return false;
                }

                if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - ccc.rho_mat_ave_compact_list[ix]) > 0.0001) {
                  printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                      cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], ccc.rho_mat_ave_compact_list[ix], i, j, mat);
                  return false;
                }
              }
            }
            else {
              // NOTE: HACK: we index materials from zero, but zero can be a list index
              int mat = ix - 1;
              if (fabs(cc.p[(i+sizex*j)*Nmats+mat] - ccc.p_compact[i+sizex*j]) > 0.0001) {
                printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                    cc.p[(i+sizex*j)*Nmats+mat], ccc.p_compact[i+sizex*j], i, j, mat);
                return false;
              }

              if (fabs(cc.rho_mat_ave[(i+sizex*j)*Nmats+mat] - ccc.rho_mat_ave_compact[i+sizex*j]) > 0.0001) {
                printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",
                    cc.rho_mat_ave[(i+sizex*j)*Nmats+mat], ccc.rho_mat_ave_compact[i+sizex*j], i, j, mat);
                return false;
              }
            }
          }
        }
        printf("All tests passed!\n");
        return true;
      }
