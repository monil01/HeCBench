#include <random>
#include <fstream>
int main(){std::default_random_engine rng;std::normal_distribution<double> dist(0.,1.);std::ofstream f("source_normals.bin",std::ios::binary);for(int i=0;i<100*32768;i++){double x=dist(rng);f.write((char*)&x,8);}}
