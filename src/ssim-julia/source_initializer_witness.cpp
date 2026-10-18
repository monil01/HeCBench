#include <cstdlib>
#include <cstdint>
#include <vector>
#include <fstream>
int main(int argc,char**argv){if(argc!=2)return 2;constexpr int count=35584;std::vector<float>x(count),y(count);srand(123);for(int i=0;i<count;i++){x[i]=rand()/(float)RAND_MAX;y[i]=x[i]*.75f;}std::ofstream f(argv[1],std::ios::binary);f.write((char*)x.data(),x.size()*4);f.write((char*)y.data(),y.size()*4);}
