#define main sph_original_cli_main
#include "../sph-cuda/fluid.cu"
#undef main
#include <fstream>
#include <cassert>
#include <string>
template<class T>void save(std::string name,T*p,size_t n){std::ofstream f(name,std::ios::binary);f.write((char*)p,n*sizeof(T));assert(f);}
int main(int argc,char**argv){assert(argc==2);std::string root=argv[1];param p{};AABB water,box;fluid_particle*f=nullptr;boundary_particle*b=nullptr;initParams(&water,&box,&p);initParticles(&f,&b,&water,&box,&p);eulerStart(f,b,&p);for(int i=0;i<p.number_fluid_particles;i++)f[i].pressure=0.;save(root+"/default_initial_f64.bin",f,p.number_fluid_particles);save(root+"/default_boundary_f64.bin",b,p.number_boundary_particles);save(root+"/default_params_raw.bin",&p,1);
param c=p;c.number_fluid_particles=1;c.number_boundary_particles=1;c.number_particles=2;c.number_steps=17;fluid_particle one{};one.density=p.rest_density;one.pos=make_double3(.25,.3,.4);boundary_particle far{};far.pos=make_double3(.25,.3,100.);far.n=make_double3(0.,0.,-1.);eulerStart(&one,&far,&c);save(root+"/ballistic_initial_f64.bin",&one,1);save(root+"/ballistic_boundary_f64.bin",&far,1);save(root+"/ballistic_params_raw.bin",&c,1);printf("GENERATED n=%d nb=%d steps=%d params=%zu fluid=%zu boundary=%zu; definedunusedinitialpressure0andparampadding0; sourcekernelsunchanged\n",p.number_fluid_particles,p.number_boundary_particles,p.number_steps,sizeof(param),sizeof(fluid_particle),sizeof(boundary_particle));finalizeParticles(f,b);}
