#include "device.hpp"

extern "C" __global__ void generate_put(int steps,int n,PayoffPut p,double dt,double s,double r,double sigma,const double* samples,double* paths){generate_paths_kernel<256>(steps,n,p,dt,s,r,sigma,samples,paths);}
extern "C" __global__ void svd_put(int n,int minimum,PayoffPut p,const double* paths,int* skip,double* svd){prepare_svd_kernel<256>(n,minimum,p,paths,skip,svd);}
extern "C" __global__ void partial_put(int n,PayoffPut p,const double* svd,const double* paths,const int* skip,double* beta,int timestep,int cash_offset){compute_partial_beta_kernel<128>(n,p,svd+16*timestep,paths+size_t(timestep)*n,paths+cash_offset,skip+timestep,beta);}
extern "C" __global__ void update_put(int n,PayoffPut p,double discount,const double* beta,double* paths,const int* skip,int timestep,int cash_offset){update_cashflow_kernel<128>(n,p,discount,beta,paths+size_t(timestep)*n,skip+timestep,paths+cash_offset);}

extern "C" __global__ void generate_call(int steps,int n,PayoffCall p,double dt,double s,double r,double sigma,const double* samples,double* paths){generate_paths_kernel<256>(steps,n,p,dt,s,r,sigma,samples,paths);}
extern "C" __global__ void svd_call(int n,int minimum,PayoffCall p,const double* paths,int* skip,double* svd){prepare_svd_kernel<256>(n,minimum,p,paths,skip,svd);}
extern "C" __global__ void partial_call(int n,PayoffCall p,const double* svd,const double* paths,const int* skip,double* beta,int timestep,int cash_offset){compute_partial_beta_kernel<128>(n,p,svd+16*timestep,paths+size_t(timestep)*n,paths+cash_offset,skip+timestep,beta);}
extern "C" __global__ void update_call(int n,PayoffCall p,double discount,const double* beta,double* paths,const int* skip,int timestep,int cash_offset){update_cashflow_kernel<128>(n,p,discount,beta,paths+size_t(timestep)*n,skip+timestep,paths+cash_offset);}

extern "C" __global__ void final_beta(const int* skip,double* beta,int t){compute_final_beta_kernel<128>(skip+t,beta);}
extern "C" __global__ void partial_sums(int n,const double* paths,double* sums,int cash_offset){compute_partial_sums_kernel<128>(n,paths+cash_offset,sums);}
extern "C" __global__ void final_sum(int n,int blocks,double discount,double* sums){compute_final_sum_kernel<128>(n,blocks,discount,sums);}
