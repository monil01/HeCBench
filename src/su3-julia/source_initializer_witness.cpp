// Exact original initializer functions extracted from su3_nn_bench.cu; LICENSE.source.
#define MILC_COMPLEX
#define PRECISION 1
#include "../su3-cuda/lattice.hpp"
#include <cstdio>
#include <cstddef>
#include <vector>
#include <cstring>
#include <fstream>
// initializes su3_matrix to a given value
void init_link(su3_matrix *s, Complx val) {
  for(int j=0; j<4; ++j) for(int k=0; k<3; ++k) for(int l=0; l<3; ++l) {
    s[j].e[k][l] = val;
  }
}

// initializes a lattice site 
void make_lattice(site *s, size_t n, Complx val) {
  int nx=n;
  int ny=n;
  int nz=n;
  int nt=n;
  for(int t=0;t<nt;t++) {
    int i=t*nz*ny*nx;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++,i++){
      s[i].x=x; s[i].y=y; s[i].z=z; s[i].t=t;
      s[i].index = x+nx*(y+ny*(z+nz*t));
      if( (x+y+z+t)%2 == 0)
        s[i].parity=EVEN;
      else
        s[i].parity=ODD;
      init_link(&s[i].link[0], val);
    }
  }
}

int main(){static_assert(sizeof(site)==320);static_assert(sizeof(su3_matrix)==72);std::vector<site>a(16);std::vector<su3_matrix>b(4);memset(a.data(),0xa5,a.size()*sizeof(site));memset(b.data(),0xa5,b.size()*sizeof(su3_matrix));make_lattice(a.data(),2,Complx{Real(1),Real(0)});init_link(b.data(),Complx{Real(1.0/3.0),Real(0)});std::ofstream("initializer_sites.bin",std::ios::binary).write((char*)a.data(),a.size()*sizeof(site));std::ofstream("initializer_B.bin",std::ios::binary).write((char*)b.data(),b.size()*sizeof(su3_matrix));printf("{\"sizeof_site\":%zu,\"alignof_site\":%zu,\"sizeof_matrix\":%zu,\"offset_link\":%zu,\"offset_x\":%zu,\"offset_y\":%zu,\"offset_z\":%zu,\"offset_t\":%zu,\"offset_index\":%zu,\"offset_parity\":%zu,\"offset_pad\":%zu}\n",sizeof(site),alignof(site),sizeof(su3_matrix),offsetof(site,link),offsetof(site,x),offsetof(site,y),offsetof(site,z),offsetof(site,t),offsetof(site,index),offsetof(site,parity),offsetof(site,pad));}
