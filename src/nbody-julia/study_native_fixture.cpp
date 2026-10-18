#include <random>
#include <vector>
#include <fstream>
#include "../nbody-cuda/Particle.hpp"
int main(int argc,char**argv){std::ofstream out(argv[1],std::ios::binary);for(int n:{33,257}){std::vector<Particle>p(n);std::mt19937 pos(42),vel(42),mass(42);std::uniform_real_distribution<float>unit(0.f,1.f),signedunit(-1.f,1.f);for(auto&x:p){for(int j=0;j<3;j++)x.pos[j]=unit(pos);for(int j=0;j<3;j++)x.vel[j]=signedunit(vel)*1e-3f;x.mass=float(n)*unit(mass);}out.write((char*)p.data(),p.size()*sizeof(Particle));}}
