#include <iris/iris.hpp>
#include <vector>
#include <cstring>
#include <ctime>
#include "constants.h"
#include "grid.h"
#define R 4
#define NDIM 8
struct dim3{size_t x,y,z;dim3(size_t a,size_t b,size_t c):x(a),y(b),z(c){}};
void target(uint nsteps, double *time_kernel,
            const struct grid_t grid,
            llint sx, llint sy, llint sz,
            float hdx_2, float hdy_2, float hdz_2,
            const float *__restrict__ coefx,
            const float *__restrict__ coefy,
            const float *__restrict__ coefz,
                  float *__restrict__ u,
            const float *__restrict__ v,
            const float *__restrict__ vp,
            const float *__restrict__ phi,
            const float *__restrict__ eta,
            const float *__restrict__ source) 
{
  struct timespec start, end;

  size_t count=gridSize(grid)/sizeof(float);
  std::vector<float> u_buf(count,0),v_buf(count,0),phi_buf(phi,phi+count);
  iris::DMem u_mem(u_buf.data(),gridSize(grid)),v_mem(v_buf.data(),gridSize(grid)),phi_mem(phi_buf.data(),gridSize(grid));
  iris::DMem vp_mem(const_cast<float*>(vp),gridSize(grid)),eta_mem(const_cast<float*>(eta),gridSize(grid));
  iris::DMem* d_u=&u_mem; iris::DMem* d_v=&v_mem;
  iris::DMem* d_phi=&phi_mem; iris::DMem* d_eta=&eta_mem; iris::DMem* d_vp=&vp_mem;

  const llint xmin = 0; const llint xmax = grid.nx;
  const llint ymin = 0; const llint ymax = grid.ny;

  dim3 threadsPerBlock(NDIM, NDIM, NDIM);

  #ifdef DEBUG
  const uint npo = 100;
  #endif

  clock_gettime(CLOCK_REALTIME, &start);

  for (uint istep = 1; istep <= nsteps; ++istep) {

      dim3 n_block_front(
          (grid.z2-grid.z1+NDIM-1) / NDIM,
          (grid.ny+NDIM-1) / NDIM,
          (grid.nx+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(xmin);
        llint arg7=(xmax);
        llint arg8=(ymin);
        llint arg9=(ymax);
        llint arg10=(grid.z1);
        llint arg11=(grid.z2);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_front.x)*NDIM,size_t(n_block_front.y)*NDIM,size_t(n_block_front.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_pml_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      dim3 n_block_top(
          (grid.z4-grid.z3+NDIM-1) / NDIM,
          (grid.y2-grid.y1+NDIM-1) / NDIM,
          (grid.nx+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(xmin);
        llint arg7=(xmax);
        llint arg8=(grid.y1);
        llint arg9=(grid.y2);
        llint arg10=(grid.z3);
        llint arg11=(grid.z4);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_top.x)*NDIM,size_t(n_block_top.y)*NDIM,size_t(n_block_top.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_pml_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      dim3 n_block_left(
          (grid.z4-grid.z3+NDIM-1) / NDIM,
          (grid.y4-grid.y3+NDIM-1) / NDIM,
          (grid.x2-grid.x1+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(grid.x1);
        llint arg7=(grid.x2);
        llint arg8=(grid.y3);
        llint arg9=(grid.y4);
        llint arg10=(grid.z3);
        llint arg11=(grid.z4);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_left.x)*NDIM,size_t(n_block_left.y)*NDIM,size_t(n_block_left.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_pml_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      dim3 n_block_center(
          (grid.z4-grid.z3+NDIM-1) / NDIM,
          (grid.y4-grid.y3+NDIM-1) / NDIM,
          (grid.x4-grid.x3+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(grid.x3);
        llint arg7=(grid.x4);
        llint arg8=(grid.y3);
        llint arg9=(grid.y4);
        llint arg10=(grid.z3);
        llint arg11=(grid.z4);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_center.x)*NDIM,size_t(n_block_center.y)*NDIM,size_t(n_block_center.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_inner_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      dim3 n_block_right(
          (grid.z4-grid.z3+NDIM-1) / NDIM,
          (grid.y4-grid.y3+NDIM-1) / NDIM,
          (grid.x6-grid.x5+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(grid.x5);
        llint arg7=(grid.x6);
        llint arg8=(grid.y3);
        llint arg9=(grid.y4);
        llint arg10=(grid.z3);
        llint arg11=(grid.z4);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_right.x)*NDIM,size_t(n_block_right.y)*NDIM,size_t(n_block_right.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_pml_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      dim3 n_block_bottom(
          (grid.z4-grid.z3+NDIM-1) / NDIM,
          (grid.y6-grid.y5+NDIM-1) / NDIM,
          (grid.nx+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(xmin);
        llint arg7=(xmax);
        llint arg8=(grid.y5);
        llint arg9=(grid.y6);
        llint arg10=(grid.z3);
        llint arg11=(grid.z4);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_bottom.x)*NDIM,size_t(n_block_bottom.y)*NDIM,size_t(n_block_bottom.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_pml_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      dim3 n_block_back(
          (grid.z6-grid.z5+NDIM-1) / NDIM,
          (grid.ny+NDIM-1) / NDIM,
          (grid.nx+NDIM-1) / NDIM);
      {
        llint arg0=(grid.nx);
        llint arg1=(grid.ny);
        llint arg2=(grid.nz);
        int arg3=(grid.ldimx);
        int arg4=(grid.ldimy);
        int arg5=(grid.ldimz);
        llint arg6=(xmin);
        llint arg7=(xmax);
        llint arg8=(ymin);
        llint arg9=(ymax);
        llint arg10=(grid.z5);
        llint arg11=(grid.z6);
        llint arg12=(grid.lx);
        llint arg13=(grid.ly);
        llint arg14=(grid.lz);
        float arg15=(hdx_2);
        float arg16=(hdy_2);
        float arg17=(hdz_2);
        float arg18=(coefx[0]+coefy[0]+coefz[0]);
        float arg19=(coefx[1]);
        float arg20=(coefx[2]);
        float arg21=(coefx[3]);
        float arg22=(coefx[4]);
        float arg23=(coefy[1]);
        float arg24=(coefy[2]);
        float arg25=(coefy[3]);
        float arg26=(coefy[4]);
        float arg27=(coefz[1]);
        float arg28=(coefz[2]);
        float arg29=(coefz[3]);
        float arg30=(coefz[4]);
        void* args[]={&arg0,&arg1,&arg2,&arg3,&arg4,&arg5,&arg6,&arg7,&arg8,&arg9,&arg10,&arg11,&arg12,&arg13,&arg14,&arg15,&arg16,&arg17,&arg18,&arg19,&arg20,&arg21,&arg22,&arg23,&arg24,&arg25,&arg26,&arg27,&arg28,&arg29,&arg30,d_u,d_v,d_vp,d_phi,d_eta};
        int info[]={sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4),sizeof(arg5),sizeof(arg6),sizeof(arg7),sizeof(arg8),sizeof(arg9),sizeof(arg10),sizeof(arg11),sizeof(arg12),sizeof(arg13),sizeof(arg14),sizeof(arg15),sizeof(arg16),sizeof(arg17),sizeof(arg18),sizeof(arg19),sizeof(arg20),sizeof(arg21),sizeof(arg22),sizeof(arg23),sizeof(arg24),sizeof(arg25),sizeof(arg26),sizeof(arg27),sizeof(arg28),sizeof(arg29),sizeof(arg30),iris_r,iris_rw,iris_r,iris_rw,iris_r};
        size_t gws[]={size_t(n_block_back.x)*NDIM,size_t(n_block_back.y)*NDIM,size_t(n_block_back.z)*NDIM},lws[]={NDIM,NDIM,NDIM};
        iris::Task task;if(task.kernel("target_pml_3d_kernel",3,nullptr,gws,lws,36,args,info)!=IRIS_SUCCESS || task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      {
        llint idx=IDX3_grid(sx,sy,sz,grid); float val=source[istep];
        iris::Task task;size_t gws[]={1},lws[]={1};
        void* args[]={d_v,&idx,&val};int info[]={iris_rw,sizeof(idx),sizeof(val)};
        if(task.kernel("kernel_add_source_kernel",1,nullptr,gws,lws,3,args,info)!=IRIS_SUCCESS) exit(2);
        if(istep==nsteps) task.flush_out(*d_v);
        if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(2);
      }

      iris::DMem *t = d_u;
      d_u = d_v;
      d_v = t;

      // Print out
      #ifdef DEBUG
      if (istep % npo == 0) printf("time step %u / %u\n", istep, nsteps);
      #endif
  }

  clock_gettime(CLOCK_REALTIME, &end);
  *time_kernel = (end.tv_sec  - start.tv_sec) +
                 (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

  memcpy(u,d_u==&u_mem ? u_buf.data() : v_buf.data(),gridSize(grid));





}
