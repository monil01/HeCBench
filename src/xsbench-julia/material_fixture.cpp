#include "../xsbench-cuda/Materials.cu"
#include <vector>
#include <fstream>
double LCG_random_double(uint64_t*seed){*seed=(2806196910506780709ULL*(*seed)+1)&0x7fffffffffffffffULL;return double(*seed)/9223372036854775808.;}
int main(int argc,char**argv){if(argc!=2)return 1;int *num=load_num_nucs(68),mx;int *mats=load_mats(num,68,&mx);std::vector<int> out{12,mx};for(int i=0;i<12;i++)out.push_back(num[i]);for(int m=0;m<12;m++)for(int j=0;j<mx;j++)out.push_back(j<num[m]?mats[m*mx+j]:-777);std::ofstream f(argv[1],std::ios::binary);f.write((char*)out.data(),out.size()*4);return !f;}
