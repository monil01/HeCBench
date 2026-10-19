#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "../gerbil-cuda/include/gerbil/KMer.h"
#ifndef GERBIL_ROOT
#define GERBIL_ROOT "."
#endif
int main(){std::ifstream f(std::string(GERBIL_ROOT)+"/probe.bin",std::ios::binary);uint32_t n,nu;f.read((char*)&n,4);f.read((char*)&nu,4);std::vector<uint64_t>keys(n);f.read((char*)keys.data(),n*8);int wrong=0;for(auto word:keys){unsigned char bytes[8];for(int i=0;i<8;i++)bytes[i]=(word>>(56-i*8))&255;const unsigned char*p=bytes;gerbil::KMer<28>forward,inverse;forward.set(p);inverse.setInv(forward);auto normalized=forward.getNormalized(inverse);if(sizeof(forward)!=8||forward.data!=word||normalized.data!=word)wrong++;}printf("SOURCE_KMER_LAYOUT {\"keys_checked\":%u,\"source_KMer28_size\":%zu,\"wrong_packing_or_canonical\":%d}\n",n,sizeof(gerbil::KMer<28>),wrong);return wrong?1:0;}
