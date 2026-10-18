#include <cstdlib>
#include <cstdio>
#include <vector>
#include <fstream>
#include "../spgemm-cuda/utils.h"
int main(int argc,char**argv){if(argc!=2)return 2;std::vector<float>a(1024*1024),b(1024*1024);init_matrix(a.data(),1024,1024,1024);init_matrix(b.data(),1024,1024,1024*1024);std::ofstream(std::string(argv[1])+"/source_A_f32.bin",std::ios::binary).write((char*)a.data(),a.size()*4);std::ofstream(std::string(argv[1])+"/source_B_f32.bin",std::ios::binary).write((char*)b.data(),b.size()*4);}
