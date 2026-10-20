#define main sph_original_cli_main
#include "../../sph-cuda/fluid.cu"
#undef main
#include <fstream>
#include <cassert>
template<class T>void load(const char*f,T*p,size_t n){std::ifstream s(f,std::ios::binary);s.read((char*)p,n*sizeof(T));assert(s);}
int main(){param p;fluid_particle f;boundary_particle b;load("ballistic_params_raw.bin",&p,1);load("ballistic_initial_f64.bin",&f,1);load("ballistic_boundary_f64.bin",&b,1);fluid_particle*df;boundary_particle*db;param*dp;cudaMalloc((void**)&df,sizeof(f));cudaMalloc((void**)&db,sizeof(b));cudaMalloc((void**)&dp,sizeof(p));cudaMemcpy(df,&f,sizeof(f),cudaMemcpyHostToDevice);cudaMemcpy(db,&b,sizeof(b),cudaMemcpyHostToDevice);cudaMemcpy(dp,&p,sizeof(p),cudaMemcpyHostToDevice);for(int n=0;n<p.number_steps;n++){updatePressures<<<1,256>>>(df,dp);updateAccelerationsFP<<<1,256>>>(df,dp);updateAccelerationsBP<<<1,256>>>(df,db,dp);updatePositions<<<1,256>>>(df,dp);}assert(cudaDeviceSynchronize()==cudaSuccess);cudaMemcpy(&f,df,sizeof(f),cudaMemcpyDeviceToHost);std::ofstream o("ballistic_actual_full_state_f64.bin",std::ios::binary);o.write((char*)&f,sizeof(f));assert(o);printf("nsteps%d dt%.17g g%.17g z%.17g vz%.17g vhz%.17g az%.17g rho%.17g p%.17g\n",p.number_steps,p.time_step,p.g,f.pos.z,f.v.z,f.v_half.z,f.a.z,f.density,f.pressure);cudaFree(df);cudaFree(db);cudaFree(dp);}
