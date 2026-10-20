#define main sph_original_cli_main
#include "../../sph-cuda/fluid.cu"
#undef main
#include <fstream>
#include <cassert>
#include <cstddef>
void check(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));exit(2);}}
template<class T>void dump(const char*name,T*p,size_t n){std::ofstream f(name,std::ios::binary);f.write((char*)p,n*sizeof(T));assert(f);}
int main(){param params;AABB water,boundary;fluid_particle*f=nullptr;boundary_particle*b=nullptr;initParams(&water,&boundary,&params);initParticles(&f,&b,&water,&boundary,&params);eulerStart(f,b,&params);fluid_particle*df;boundary_particle*db;param*dp;int n=params.number_fluid_particles,nb=params.number_boundary_particles;check(cudaMalloc((void**)&df,n*sizeof(*df)));check(cudaMalloc((void**)&db,nb*sizeof(*db)));check(cudaMalloc((void**)&dp,sizeof(param)));check(cudaMemcpy(df,f,n*sizeof(*df),cudaMemcpyHostToDevice));check(cudaMemcpy(db,b,nb*sizeof(*db),cudaMemcpyHostToDevice));check(cudaMemcpy(dp,&params,sizeof(param),cudaMemcpyHostToDevice));
check(cudaMemcpy(f,df,n*sizeof(*df),cudaMemcpyDeviceToHost));check(cudaMemcpy(b,db,nb*sizeof(*db),cudaMemcpyDeviceToHost));check(cudaMemcpy(&params,dp,sizeof(param),cudaMemcpyDeviceToHost));dump("original_GPU_initial_fluid_raw.bin",f,n);dump("original_GPU_boundary_raw.bin",b,nb);dump("original_GPU_params_raw.bin",&params,1);
for(int step=0;step<params.number_steps;step++){updatePressures<<<(n+255)/256,256>>>(df,dp);updateAccelerationsFP<<<(n+255)/256,256>>>(df,dp);updateAccelerationsBP<<<(nb+255)/256,256>>>(df,db,dp);updatePositions<<<(n+255)/256,256>>>(df,dp);}check(cudaDeviceSynchronize());check(cudaMemcpy(f,df,n*sizeof(*df),cudaMemcpyDeviceToHost));dump("original_GPU_final_fluid_raw.bin",f,n);printf("CAPTURE n=%d nb=%d steps=%d fluid_size=%zu boundary_size=%zu param_size=%zu pos_offset=%zu v_offset=%zu half_offset=%zu a_offset=%zu\n",n,nb,params.number_steps,sizeof(fluid_particle),sizeof(boundary_particle),sizeof(param),offsetof(fluid_particle,pos),offsetof(fluid_particle,v),offsetof(fluid_particle,v_half),offsetof(fluid_particle,a));check(cudaFree(df));check(cudaFree(db));check(cudaFree(dp));finalizeParticles(f,b);}
