#include <cuda_runtime.h>
#include "kernels.h"
extern "C" __global__ void make_flag(flag_type* flags,int* values,flag_type* bd,BoxCU domain,outer_wall wall,const char* priority){
 make_flag_impl(flags,values,bd,domain,wall,domain.nx,domain.ny,domain.nz,0,priority);
}
extern "C" __global__ void find_wall(flag_type* flags,flag_type* bd,int* values,BoxCU domain,const char* directions){
 find_wall_impl<D3Q19>(flags,bd,values,domain,0,directions);
}
extern "C" __global__ void init_velocity(float* velocity,double* rho,double* f0,double* f1,BoxCU domain){
 const int n=domain.nx*domain.ny*domain.nz;lbm_vars v{};
 v.u_star={velocity,velocity+n,velocity+2*n};v.r=rho;v.f0=f0;v.f1=f1;
 init_velocity_g_impl<D3Q19>(v,domain,domain,domain.nz,0,0,0,1.);
}
extern "C" __global__ __launch_bounds__(64) void collide(double* f0,double* f1,float* velocity,flag_type* flags,flag_type* bd,int* values,BoxCU domain,double ulb,double omega,bool out_u,int iter){
 const int n=domain.nx*domain.ny*domain.nz;lbm_vars v{};
 v.u_star={velocity,velocity+n,velocity+2*n};v.f0=f0;v.f1=f1;v.boundary_flag=flags;v.boundary_dirs=bd;v.boundary_values=values;
 collide_and_stream_g_impl<D3Q19>(v,domain,ulb,omega,out_u,iter);
}
