#pragma once
#include "binding_types.h"
extern "C" __global__ void egs_bind_core(MT_tables_t* mtt,MT_param* mtp,uint* mts,uint4* sa,uint4* sb,uint4* sc,total_step_counts_t* steps,total_list_depth_t* depth,total_num_inner_iterations_t* iters,total_weights_t* weights,region_data_t* regions,float2* ge_,float2* gmfp_,float2* gbr1_,float2* gbr2_,float2* cohe_,float2* pmax_,rayleigh_data_t* ray,uint* ia){
  MT_tables=mtt;
  MT_params=mtp;
  MT_statuses=mts;
  stack.a=sa;
  stack.b=sb;
  stack.c=sc;
  total_step_counts=steps;
  total_list_depth=depth;
  total_num_inner_iterations=iters;
  total_weights=weights;
  region_data=regions;
  ge=ge_;
  gmfp=gmfp_;
  gbr1=gbr1_;
  gbr2=gbr2_;
  cohe=cohe_;
  pmax=pmax_;
  rayleigh_data=ray;
  i_array=ia;
}

extern "C" __global__ void egs_probe_bindings(unsigned long* out,MT_tables_t* mtt,MT_param* mtp,uint* mts,uint4* sa,uint4* sb,uint4* sc,total_step_counts_t* steps,total_list_depth_t* depth,total_num_inner_iterations_t* iters,total_weights_t* weights,region_data_t* regions,float2* ge_,float2* gmfp_,float2* gbr1_,float2* gbr2_,float2* cohe_,float2* pmax_,rayleigh_data_t* ray,uint* ia){
  out[0]=(MT_tables==mtt);
  out[1]=(MT_params==mtp);
  out[2]=(MT_statuses==mts);
  out[3]=(stack.a==sa);
  out[4]=(stack.b==sb);
  out[5]=(stack.c==sc);
  out[6]=(total_step_counts==steps);
  out[7]=(total_list_depth==depth);
  out[8]=(total_num_inner_iterations==iters);
  out[9]=(total_weights==weights);
  out[10]=(region_data==regions);
  out[11]=(ge==ge_);
  out[12]=(gmfp==gmfp_);
  out[13]=(gbr1==gbr1_);
  out[14]=(gbr2==gbr2_);
  out[15]=(cohe==cohe_);
  out[16]=(pmax==pmax_);
  out[17]=(rayleigh_data==ray);
  out[18]=(i_array==ia);
  out[19]=sizeof(source_t);out[20]=sizeof(detector_t);out[21]=sizeof(phantom_t);
  out[22]=detector.N.x;out[23]=detector.N.y;out[24]=phantom.N.x;out[25]=phantom.N.y;out[26]=phantom.N.z;out[27]=source.n;
  out[28]=MT_params[0].M;out[29]=MT_statuses[0];
  printf("GPU source/phantom pointers xi=%p wi=%p bin=%p bounds=%p,%p,%p scores=%p,%p totals=%p\n",source.xi,source.wi,source.bin,phantom.x_bounds,phantom.y_bounds,phantom.z_bounds,detector_scores_count[0][0],detector_scores_energy[0][0],detector_totals_count[0]);
}
extern "C" __global__ void egs_bind_source(const source_bind_t* meta,float* xi,float* wi,int* bin){
  source.n=meta->n;source.xi=xi;source.wi=wi;source.bin=bin;
  source.source_point=make_float3(meta->px,meta->py,meta->pz);source.rectangle_z=meta->z;
  source.rectangle_min=make_float2(meta->minx,meta->miny);source.rectangle_max=make_float2(meta->maxx,meta->maxy);
  source.rectangle_size=make_float2(meta->sizex,meta->sizey);source.rectangle_area=meta->area;
}
extern "C" __global__ void egs_bind_phantom(const phantom_bind_t* meta,float* x,float* y,float* z){
  phantom.N=make_uint3(meta->nx,meta->ny,meta->nz);phantom.x_bounds=x;phantom.y_bounds=y;phantom.z_bounds=z;
}
extern "C" __global__ void egs_bind_detector(const detector_bind_t* meta){
  detector.center=make_float3(meta->x,meta->y,meta->z);detector.d=make_float2(meta->dx,meta->dy);detector.N=make_uint2(meta->nx,meta->ny);
}
extern "C" __global__ void egs_bind_scores(float* count,float* energy,int block,int cat){
  detector_scores_count[block][cat]=count;detector_scores_energy[block][cat]=energy;
}
extern "C" __global__ void egs_bind_totals(double* count,double* energy,int cat){
  detector_totals_count[cat]=count;detector_totals_energy[cat]=energy;
}
