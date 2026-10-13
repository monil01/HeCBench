#pragma once
#include "memory.h"
struct Dim3 { size_t x,y,z; Dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c) {} };
inline void launch_init_facexy(Dim3 grid,Dim3 block,int ix_base,int iy_base,int dims_b_ne,int dims_b_na,int dims_b_ncell_x,int dims_b_ncell_y,int dims_b_ncell_z,int dims_ncell_z,iris::DMem *facexy) {
 iris::Task task; size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={&ix_base,&iy_base,&dims_b_ne,&dims_b_na,&dims_b_ncell_x,&dims_b_ncell_y,&dims_b_ncell_z,&dims_ncell_z,facexy};
 int info[]={sizeof(ix_base),sizeof(iy_base),sizeof(dims_b_ne),sizeof(dims_b_na),sizeof(dims_b_ncell_x),sizeof(dims_b_ncell_y),sizeof(dims_b_ncell_z),sizeof(dims_ncell_z),iris_rw};
 task.kernel("init_facexy",3,nullptr,g,l,9,p,info);
 task.flush_out(*facexy);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_init_facexz(Dim3 grid,Dim3 block,int ix_base,int iy_base,int dims_b_ne,int dims_b_na,int dims_b_ncell_x,int dims_b_ncell_y,int dims_b_ncell_z,int proc_y_min,int proc_y_max,StepInfoAll stepinfoall,iris::DMem *facexz) {
 iris::Task task; size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={&ix_base,&iy_base,&dims_b_ne,&dims_b_na,&dims_b_ncell_x,&dims_b_ncell_y,&dims_b_ncell_z,&proc_y_min,&proc_y_max,&stepinfoall,facexz};
 int info[]={sizeof(ix_base),sizeof(iy_base),sizeof(dims_b_ne),sizeof(dims_b_na),sizeof(dims_b_ncell_x),sizeof(dims_b_ncell_y),sizeof(dims_b_ncell_z),sizeof(proc_y_min),sizeof(proc_y_max),sizeof(stepinfoall),iris_rw};
 task.kernel("init_facexz",3,nullptr,g,l,11,p,info);
 task.flush_out(*facexz);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_init_faceyz(Dim3 grid,Dim3 block,int ix_base,int iy_base,int dims_b_ne,int dims_b_na,int dims_b_ncell_x,int dims_b_ncell_y,int dims_b_ncell_z,int proc_x_min,int proc_x_max,StepInfoAll stepinfoall,iris::DMem *faceyz) {
 iris::Task task; size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={&ix_base,&iy_base,&dims_b_ne,&dims_b_na,&dims_b_ncell_x,&dims_b_ncell_y,&dims_b_ncell_z,&proc_x_min,&proc_x_max,&stepinfoall,faceyz};
 int info[]={sizeof(ix_base),sizeof(iy_base),sizeof(dims_b_ne),sizeof(dims_b_na),sizeof(dims_b_ncell_x),sizeof(dims_b_ncell_y),sizeof(dims_b_ncell_z),sizeof(proc_x_min),sizeof(proc_x_max),sizeof(stepinfoall),iris_rw};
 task.kernel("init_faceyz",3,nullptr,g,l,11,p,info);
 task.flush_out(*faceyz);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_wavefronts(Dim3 grid,Dim3 block,int num_wavefronts,int ix_base,int iy_base,int v_b_size,int noctant_per_block,Dimensions dims_b,StepInfoAll stepinfoall,iris::DMem *facexy,iris::DMem *facexz,iris::DMem *faceyz,iris::DMem *a_from_m,iris::DMem *m_from_a,iris::DMem *vi,iris::DMem *vo,iris::DMem *vs_local) {
 iris::Task task; size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={&num_wavefronts,&ix_base,&iy_base,&v_b_size,&noctant_per_block,&dims_b,&stepinfoall,facexy,facexz,faceyz,a_from_m,m_from_a,vi,vo,vs_local};
 int info[]={sizeof(num_wavefronts),sizeof(ix_base),sizeof(iy_base),sizeof(v_b_size),sizeof(noctant_per_block),sizeof(dims_b),sizeof(stepinfoall),iris_rw,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_rw,iris_rw};
 task.kernel("wavefronts",3,nullptr,g,l,15,p,info);
 task.flush_out(*facexy); task.flush_out(*facexz); task.flush_out(*faceyz); task.flush_out(*vo); task.flush_out(*vs_local);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
