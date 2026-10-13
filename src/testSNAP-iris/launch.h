#pragma once
#include <iris/iris.hpp>
struct Dim3 { size_t x,y,z; Dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c) {} };
inline void launch_reset_ulisttot(Dim3 grid,Dim3 block,iris::DMem *ulisttot,int ulisttot_size) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={ulisttot,&ulisttot_size};
 int info[]={iris_w,sizeof(ulisttot_size)};
 task.kernel("reset_ulisttot",3,nullptr,g,l,2,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_set_ulisttot(Dim3 grid,Dim3 block,iris::DMem *ulisttot,iris::DMem *idxu_block,int num_atoms,int twojmax,double wself) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={ulisttot,idxu_block,&num_atoms,&twojmax,&wself};
 int info[]={iris_rw,iris_r,sizeof(num_atoms),sizeof(twojmax),sizeof(wself)};
 task.kernel("set_ulisttot",3,nullptr,g,l,5,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_update_ulisttot(Dim3 grid,Dim3 block,iris::DMem *rij,iris::DMem *rcutij,iris::DMem *wj,iris::DMem *ulist_parity,iris::DMem *idxu_block,iris::DMem *rootpqarray,iris::DMem *ulist,iris::DMem *ulisttot,int num_atoms,int num_nbor,int switch_flag,int twojmax,int jdimpq) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,&num_atoms,&num_nbor,&switch_flag,&twojmax,&jdimpq};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,sizeof(num_atoms),sizeof(num_nbor),sizeof(switch_flag),sizeof(twojmax),sizeof(jdimpq)};
 task.kernel("update_ulisttot",3,nullptr,g,l,13,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_reset_ylist(Dim3 grid,Dim3 block,iris::DMem *ylist,int ylist_size) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={ylist,&ylist_size};
 int info[]={iris_w,sizeof(ylist_size)};
 task.kernel("reset_ylist",3,nullptr,g,l,2,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_compute_yi(Dim3 grid,Dim3 block,iris::DMem *idxz,iris::DMem *idxzbeta,iris::DMem *cglist,iris::DMem *idxcg_block,iris::DMem *idxu_block,iris::DMem *idxdu_block,iris::DMem *ulisttot,iris::DMem *ylist,int num_atoms,int idxz_max,int jdim) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,&num_atoms,&idxz_max,&jdim};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(num_atoms),sizeof(idxz_max),sizeof(jdim)};
 task.kernel("compute_yi",3,nullptr,g,l,11,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_compute_duidrj(Dim3 grid,Dim3 block,iris::DMem *wj,iris::DMem *rij,iris::DMem *rcutij,iris::DMem *rootpqarray,iris::DMem *ulist,iris::DMem *dulist,int num_atoms,int num_nbor,int twojmax,int idxdu_max,int jdimpq,int switch_flag) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={wj,rij,rcutij,rootpqarray,ulist,dulist,&num_atoms,&num_nbor,&twojmax,&idxdu_max,&jdimpq,&switch_flag};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,sizeof(num_atoms),sizeof(num_nbor),sizeof(twojmax),sizeof(idxdu_max),sizeof(jdimpq),sizeof(switch_flag)};
 task.kernel("compute_duidrj",3,nullptr,g,l,12,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_compute_deidrj(Dim3 grid,Dim3 block,iris::DMem *idxdu_block,iris::DMem *dulist,iris::DMem *ylist,iris::DMem *dedr,int num_atoms,int num_nbor,int twojmax,int idxdu_max) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={idxdu_block,dulist,ylist,dedr,&num_atoms,&num_nbor,&twojmax,&idxdu_max};
 int info[]={iris_r,iris_r,iris_r,iris_w,sizeof(num_atoms),sizeof(num_nbor),sizeof(twojmax),sizeof(idxdu_max)};
 task.kernel("compute_deidrj",3,nullptr,g,l,8,p,info);
 task.flush_out(*dedr);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
