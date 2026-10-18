#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include "../simpleSpmv-cuda/utils.cpp"
template<class T>void put(std::ofstream&f,const std::vector<T>&a){f.write((const char*)a.data(),a.size()*sizeof(T));}
int main(){std::ofstream f("fixture.bin",std::ios::binary);uint32_t count=5;f.write((char*)&count,4);int ns[]={512,512,512,512,512},av[]={3,7,15,31,63};for(int z=0;z<5;z++){size_t n=ns[z],nz=n*av[z];std::vector<float>mat(n*n),x(n),val(nz);std::vector<size_t>row(n+1),col(nz);srand48(4096);init_matrix(mat.data(),n,nz);init_vector(x.data(),n);init_csr(row.data(),val.data(),col.data(),mat.data(),n,nz);uint32_t hdr[]={(uint32_t)n,(uint32_t)nz,(uint32_t)(1<<(z+1))};f.write((char*)hdr,12);put(f,mat);put(f,x);put(f,row);put(f,col);put(f,val);}}
