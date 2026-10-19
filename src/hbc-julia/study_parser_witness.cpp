#include "../hbc-cuda/parse.h"
#include <fstream>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
int main(int argc,char**argv){if(argc!=2)return 2;std::string root=argv[1];std::ifstream f(root+"/fixtures.bin",std::ios::binary);auto read=[&](){int v;if(!f.read((char*)&v,4))throw std::runtime_error("fixture bounds");return v;};int nc=read();const char*files[]={"path33.txt","grid35.txt","star1057.txt","dataset_readiness/breast_coexpress.txt"};for(int k=0;k<nc;k++){int n=read(),m=read();std::vector<int>R(n+1),C(2*m),F(2*m);for(auto&v:R)v=read();for(auto&v:C)v=read();for(auto&v:F)v=read();std::string name=root+"/"+files[k];graph g=parse(name.data());bool ok=g.n==n&&g.m==m&&std::equal(R.begin(),R.end(),g.R)&&std::equal(C.begin(),C.end(),g.C)&&std::equal(F.begin(),F.end(),g.F);printf("case%d n%d m%d original_parser_full_CSR_exact %s\n",k,n,m,ok?"PASS":"FAIL");delete[]g.R;delete[]g.C;delete[]g.F;if(!ok)return 1;}return 0;}
