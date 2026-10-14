#include <iris/iris.hpp>
#include <stdexcept>
#define main mcmd_original_main
#include "../mcmd-cuda/main.cpp"
#undef main
void force_kernel(const int total_atoms,const int block_size,const int pform,
 const double cutoff,const double ewald_alpha,const int ewald_kmax,
 const int kspace_option,const double polar_damp,const double* h_basis,
 const double* h_rbasis,d_atom* h_atom_list) {
 iris::DMem atoms(h_atom_list,sizeof(d_atom)*total_atoms);
 iris::DMem basis((void*)h_basis,9*sizeof(double)),rbasis((void*)h_rbasis,9*sizeof(double));
 iris::Task task;
 void* args[]={&atoms,(void*)&total_atoms,(void*)&cutoff,&basis,&rbasis,
  (void*)&pform,(void*)&ewald_alpha,(void*)&ewald_kmax,(void*)&kspace_option,(void*)&polar_damp};
 int info[]={iris_rw,sizeof(total_atoms),sizeof(cutoff),iris_r,iris_r,sizeof(pform),
  sizeof(ewald_alpha),sizeof(ewald_kmax),sizeof(kspace_option),sizeof(polar_damp)};
 size_t global[]={size_t((total_atoms+block_size-1)/block_size)*block_size},local[]={size_t(block_size)};
 task.kernel("calculateForceKernel",1,nullptr,global,local,10,args,info);
 task.flush_out(atoms);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) throw std::runtime_error("MCMD HIP force task failed");
}
int main(int argc,char** argv) {
 if(argc!=2) {std::cerr<<"Usage: "<<argv[0]<<" <input file>\\n";return 1;}
 iris::Platform platform; platform.init(&argc,&argv,1);
 int result=mcmd_original_main(argc,argv);platform.finalize();return result;
}
