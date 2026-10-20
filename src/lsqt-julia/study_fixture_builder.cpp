// Source-defined input preparation only, outside measured solver regions.
// Original GPUQT model implementation and seeded C++ RNG retained unchanged.
#include "original_native/model.h"
#include "original_native/vector.h"
#include "original_native/hamiltonian.h"
#include <fstream>
#include <stdexcept>
#include <cstdio>
template<class T>void write(std::ofstream&f,const T*p,size_t n){f.write((const char*)p,n*sizeof(T));if(!f)throw std::runtime_error("fixture write");}
int main(int argc,char**argv){if(argc!=3)return 2;Model model(argv[1]);int n=model.number_of_atoms;int flags[]={int(model.calculate_vac0),int(model.calculate_vac),int(model.calculate_msd),int(model.calculate_spin),int(model.calculate_ldos)};int h[]={n,model.max_neighbor,model.number_of_energy_points,model.number_of_steps_correlation,model.number_of_moments,model.number_of_random_vectors,flags[0],flags[1],flags[2],flags[3],flags[4],model.number_of_local_orbitals};std::ofstream f(argv[2],std::ios::binary);write(f,h,12);double scalar[]={model.volume,model.energy_max};write(f,scalar,2);write(f,model.energy,model.number_of_energy_points);if(model.number_of_steps_correlation)write(f,model.time_step,model.number_of_steps_correlation);write(f,model.local_orbitals.data(),model.local_orbitals.size());write(f,model.neighbor_number,n);write(f,model.neighbor_list,model.number_of_pairs);write(f,model.potential,n);write(f,model.hopping_real,model.number_of_pairs);write(f,model.hopping_imag,model.number_of_pairs);write(f,model.xx,model.number_of_pairs);Vector state(n);for(int r=0;r<model.number_of_random_vectors;r++){model.initialize_state(state,-1);write(f,state.real_part,n);write(f,state.imag_part,n);}f.close();Hamiltonian cleanup(model);printf("Source-defined fixture n=%d completed\n",n);}
