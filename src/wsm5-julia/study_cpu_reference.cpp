#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
using std::abs;
struct Geometry{unsigned x,y,z;};static Geometry blockIdx{0,0,0},blockDim{8,8,1},threadIdx{0,0,0};
float rsqrtf(float x){return 1.f/std::sqrt(x);}
#define __global__
#define MKX 4
#include "study_native_kernel.h"
using U=unsigned;
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("fixture output args");std::ifstream f(argv[1],std::ios::binary);std::ofstream o(argv[2],std::ios::binary);U nc;f.read((char*)&nc,4);if(!f||nc!=3)throw std::runtime_error("case count");for(U ci=0;ci<nc;ci++){U nx,ny;float dt;f.read((char*)&nx,4);f.read((char*)&ny,4);f.read((char*)&dt,4);std::vector<std::vector<float>>a;for(int k=0;k<15;k++){size_t n=nx*ny*(k<10?3:1);a.emplace_back(n);f.read((char*)a.back().data(),n*4);if(!f)throw std::runtime_error("short fixture");}for(U j=0;j<ny;j++)for(U i=0;i<nx;i++){blockIdx={i/8,j/8,0};threadIdx={i%8,j%8,0};wsm(a[0].data(),a[1].data(),a[2].data(),a[3].data(),a[4].data(),a[5].data(),a[6].data(),a[7].data(),a[8].data(),a[9].data(),a[10].data(),a[11].data(),a[12].data(),a[13].data(),a[14].data(),dt,1,nx,1,ny,1,3,1,nx,1,ny,1,3,1,nx,1,ny,1,3);}for(auto&v:a){for(float x:v)if(!std::isfinite(x))throw std::runtime_error("nonfinitecorrectedCPUstate");o.write((char*)v.data(),v.size()*4);}std::cout<<"CPU_SOURCE_CASE "<<ci<<" nx="<<nx<<" ny="<<ny<<" complete15fields finite\n";}if(f.peek()!=EOF)throw std::runtime_error("trailingfixture");if(!o)throw std::runtime_error("output write");return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
