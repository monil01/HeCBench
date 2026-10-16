#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#define main legacy_main
#include "../burger-cuda/main.cu"
#undef main
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);}}while(0)
int main(){int x_points=256,y_points=192,num_itrs=20,n=x_points*y_points;double del_x=2.0/(x_points-1),del_y=2.0/(y_points-1),nu=.01,del_t=.0009*del_x*del_y/nu;std::vector<double>u0(n,1),v0(n,1),uref,vref,unref,vnref,got(2*n);for(int i=0;i<y_points;i++)for(int j=0;j<x_points;j++)if(j*del_x>.5&&j*del_x<1&&i*del_y>.5&&i*del_y<1)u0[idx(i,j)]=v0[idx(i,j)]=2;uref=u0;vref=v0;unref=u0;vnref=v0;double *u=uref.data(),*v=vref.data(),*u_new=unref.data(),*v_new=vnref.data();
  for(int itr = 0; itr < num_itrs; itr++){

    for(int i = 1; i < y_points-1; i++){
      for(int j = 1; j < x_points-1; j++){
        u_new[idx(i,j)] = u[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
          (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
          (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
          (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

        v_new[idx(i,j)] = v[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
          (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
          (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
          (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
      }
    }

    // Boundary conditions
    for(int i = 0; i < x_points; i++){
      u_new[idx(0,i)] = 1.0;
      v_new[idx(0,i)] = 1.0;
      u_new[idx(y_points-1,i)] = 1.0;
      v_new[idx(y_points-1,i)] = 1.0;
    }

    for(int j = 0; j < y_points; j++){
      u_new[idx(j,0)] = 1.0;
      v_new[idx(j,0)] = 1.0;
      u_new[idx(j,x_points-1)] = 1.0;
      v_new[idx(j,x_points-1)] = 1.0;
    }

    // Updating older values to newer ones
    for(int i = 0; i < y_points; i++){
      for(int j = 0; j < x_points; j++){
        u[idx(i,j)] = u_new[idx(i,j)];
        v[idx(i,j)] = v_new[idx(i,j)];
      }
    }
  }

double *du,*dv,*un,*vn;for(auto ptr:{&du,&dv,&un,&vn})CK(cudaMalloc(ptr,n*8));auto reset=[&](){CK(cudaMemcpy(du,u0.data(),n*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,v0.data(),n*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(un,u0.data(),n*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(vn,v0.data(),n*8,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());};auto step=[&](){core<<<dim3((x_points-2+15)/16,(y_points-2+15)/16),dim3(16,16)>>>(un,vn,du,dv,x_points,y_points,nu,del_t,del_x,del_y);bound_h<<<1,256>>>(un,vn,x_points,y_points);bound_v<<<1,256>>>(un,vn,x_points,y_points);update<<<(n+255)/256,256>>>(du,dv,un,vn,n);};reset();step();CK(cudaDeviceSynchronize());bool all=true;for(int trial=1;trial<=5;trial++){reset();auto t0=std::chrono::steady_clock::now();for(int k=0;k<num_itrs;k++)step();CK(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/num_itrs;CK(cudaMemcpy(got.data(),du,n*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data()+n,dv,n*8,cudaMemcpyDeviceToHost));double ma=0,mr=0;bool ok=true;for(int i=0;i<2*n;i++){double ref=i<n?uref[i]:vref[i-n],e=fabs(got[i]-ref);ma=std::max(ma,e);mr=std::max(mr,e/std::max(fabs(ref),2.220446049250313e-16));ok&=std::isfinite(got[i])&&e<=1e-6;}all&=ok;char file[100];sprintf(file,"burger_trial%d.bin",trial);FILE*f=fopen(file,"wb");fwrite(got.data(),8,2*n,f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"four_kernel_timestep_per_step\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",2*n,ma,mr,num_itrs,file,2*n);}for(auto ptr:{du,dv,un,vn})CK(cudaFree(ptr));puts(all?"PASS":"FAIL");return all?0:1;}
