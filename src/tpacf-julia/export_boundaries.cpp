#include <fstream>
#define main original_tpacf_main
#include "corrected_native/main.cu"
#undef main
int main(int argc,char**argv){if(argc!=2)return 2;int nb=0;double*b=init_bins(5,1.f,10000.f,1,&nb);std::ofstream f(argv[1],std::ios::binary);f.write((char*)b,(NUMBINS-1)*sizeof(double));free(b);return !f;}
