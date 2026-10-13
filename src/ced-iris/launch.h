#pragma once
#include <iris/iris.hpp>
struct Dim3 {size_t x,y;Dim3(size_t a,size_t b):x(a),y(b){}};
inline void launch_gaussian(Dim3 grid,Dim3 block,iris::DMem &data,iris::DMem &out,int rows,int cols) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y},l[]={block.x,block.y};
 void *p[]={&data,&out,&rows,&cols};int info[]={iris_r,iris_rw,sizeof(rows),sizeof(cols)};
 task.kernel("gaussian_kernel",2,nullptr,g,l,4,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_sobel(Dim3 grid,Dim3 block,iris::DMem &data,iris::DMem &out,iris::DMem &theta,int rows,int cols) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y},l[]={block.x,block.y};
 void *p[]={&data,&out,&theta,&rows,&cols};int info[]={iris_r,iris_rw,iris_rw,sizeof(rows),sizeof(cols)};
 task.kernel("sobel_kernel",2,nullptr,g,l,5,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_nonmax(Dim3 grid,Dim3 block,iris::DMem &data,iris::DMem &out,iris::DMem &theta,int rows,int cols) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y},l[]={block.x,block.y};
 void *p[]={&data,&out,&theta,&rows,&cols};int info[]={iris_r,iris_rw,iris_r,sizeof(rows),sizeof(cols)};
 task.kernel("non_max_supp_kernel",2,nullptr,g,l,5,p,info);
 
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_hyst(Dim3 grid,Dim3 block,iris::DMem &data,iris::DMem &out,int rows,int cols) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y},l[]={block.x,block.y};
 void *p[]={&data,&out,&rows,&cols};int info[]={iris_r,iris_rw,sizeof(rows),sizeof(cols)};
 task.kernel("hyst_kernel",2,nullptr,g,l,4,p,info);
 task.flush_out(out);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
