#pragma once
#include "memory.h"
struct Dim3 { size_t x,y,z; Dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c) {} };
inline void launch_compute_flux_x(Dim3 grid,Dim3 block,iris::DMem *state,iris::DMem *flux,iris::DMem *hy_dens_cell,iris::DMem *hy_dens_theta_cell,double hv_coef,int nx,int nz,int hs) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,flux,hy_dens_cell,hy_dens_theta_cell,&hv_coef,&nx,&nz,&hs};
 int info[]={iris_r,iris_rw,iris_r,iris_r,sizeof(hv_coef),sizeof(nx),sizeof(nz),sizeof(hs)};
 task.kernel("compute_flux_x",3,nullptr,g,l,8,p,info);
 task.flush_out(*flux);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_compute_tend_x(Dim3 grid,Dim3 block,iris::DMem *flux,iris::DMem *tend,int nx,int nz,int dx) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={flux,tend,&nx,&nz,&dx};
 int info[]={iris_r,iris_rw,sizeof(nx),sizeof(nz),sizeof(dx)};
 task.kernel("compute_tend_x",3,nullptr,g,l,5,p,info);
 task.flush_out(*tend);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_compute_flux_z(Dim3 grid,Dim3 block,iris::DMem *state,iris::DMem *flux,iris::DMem *hy_dens_int,iris::DMem *hy_pressure_int,iris::DMem *hy_dens_theta_int,double hv_coef,int nx,int nz,int hs) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,flux,hy_dens_int,hy_pressure_int,hy_dens_theta_int,&hv_coef,&nx,&nz,&hs};
 int info[]={iris_r,iris_rw,iris_rw,iris_rw,iris_rw,sizeof(hv_coef),sizeof(nx),sizeof(nz),sizeof(hs)};
 task.kernel("compute_flux_z",3,nullptr,g,l,9,p,info);
 task.flush_out(*flux); task.flush_out(*hy_dens_int); task.flush_out(*hy_pressure_int); task.flush_out(*hy_dens_theta_int);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_compute_tend_z(Dim3 grid,Dim3 block,iris::DMem *state,iris::DMem *flux,iris::DMem *tend,int nx,int nz,int dz) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,flux,tend,&nx,&nz,&dz};
 int info[]={iris_r,iris_r,iris_rw,sizeof(nx),sizeof(nz),sizeof(dz)};
 task.kernel("compute_tend_z",3,nullptr,g,l,6,p,info);
 task.flush_out(*tend);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_pack_send_buf(Dim3 grid,Dim3 block,iris::DMem *state,iris::DMem *sendbuf_l,iris::DMem *sendbuf_r,int nx,int nz,int hs) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,sendbuf_l,sendbuf_r,&nx,&nz,&hs};
 int info[]={iris_r,iris_rw,iris_rw,sizeof(nx),sizeof(nz),sizeof(hs)};
 task.kernel("pack_send_buf",3,nullptr,g,l,6,p,info);
 task.flush_out(*sendbuf_l); task.flush_out(*sendbuf_r);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_unpack_recv_buf(Dim3 grid,Dim3 block,iris::DMem *state,iris::DMem *recvbuf_l,iris::DMem *recvbuf_r,int nx,int nz,int hs) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,recvbuf_l,recvbuf_r,&nx,&nz,&hs};
 int info[]={iris_rw,iris_r,iris_r,sizeof(nx),sizeof(nz),sizeof(hs)};
 task.kernel("unpack_recv_buf",3,nullptr,g,l,6,p,info);
 task.flush_out(*state);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_update_state_x(Dim3 grid,Dim3 block,iris::DMem *state,iris::DMem *hy_dens_cell,iris::DMem *hy_dens_theta_cell,int nx,int nz,int hs,int k_beg,double dz) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,hy_dens_cell,hy_dens_theta_cell,&nx,&nz,&hs,&k_beg,&dz};
 int info[]={iris_rw,iris_r,iris_r,sizeof(nx),sizeof(nz),sizeof(hs),sizeof(k_beg),sizeof(dz)};
 task.kernel("update_state_x",3,nullptr,g,l,8,p,info);
 task.flush_out(*state);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_update_state_z(Dim3 grid,Dim3 block,iris::DMem *state,int data_spec_int,int i_beg,int nx,int nz,int hs,double dx,double mnt_width) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state,&data_spec_int,&i_beg,&nx,&nz,&hs,&dx,&mnt_width};
 int info[]={iris_rw,sizeof(data_spec_int),sizeof(i_beg),sizeof(nx),sizeof(nz),sizeof(hs),sizeof(dx),sizeof(mnt_width)};
 task.kernel("update_state_z",3,nullptr,g,l,8,p,info);
 task.flush_out(*state);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_acc_mass_te(Dim3 grid,Dim3 block,iris::DMem *mass,iris::DMem *te,iris::DMem *state,iris::DMem *hy_dens_cell,iris::DMem *hy_dens_theta_cell,int nx,int nz,double dx,double dz) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={mass,te,state,hy_dens_cell,hy_dens_theta_cell,&nx,&nz,&dx,&dz};
 int info[]={iris_rw,iris_rw,iris_r,iris_r,iris_r,sizeof(nx),sizeof(nz),sizeof(dx),sizeof(dz)};
 task.kernel("acc_mass_te",3,nullptr,g,l,9,p,info);
 task.flush_out(*mass); task.flush_out(*te);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_update_fluid_state(Dim3 grid,Dim3 block,iris::DMem *state_init,iris::DMem *state_out,iris::DMem *tend,int nx,int nz,int hs,double dt) {
 iris::Task task;
 size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={state_init,state_out,tend,&nx,&nz,&hs,&dt};
 int info[]={iris_r,iris_rw,iris_r,sizeof(nx),sizeof(nz),sizeof(hs),sizeof(dt)};
 task.kernel("update_fluid_state",3,nullptr,g,l,7,p,info);
 task.flush_out(*state_out);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
