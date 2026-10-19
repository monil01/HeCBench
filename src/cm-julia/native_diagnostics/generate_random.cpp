#include <random>
#include <fstream>
#include <vector>
#include <cstdlib>
int main(int argc,char**argv){int R=atoi(argv[1]),sig=31;std::default_random_engine generator(123);std::uniform_real_distribution<float> distribution(0.f,1.f);for(int k=0;k<2;k++){std::vector<float>a(R*sig);for(float&v:a)v=distribution(generator);std::ofstream f(std::string(argv[2])+"/random"+std::to_string(k)+"_f32.bin",std::ios::binary);f.write((char*)a.data(),a.size()*4);if(!f)return 1;}return 0;}
