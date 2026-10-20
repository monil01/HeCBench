// CPU-only source VCD/header decoder. No Simulator constructed or GPU method called.
#include "interParser.h"
#include "vcdParser.h"
#include "FileReader.h"
#include <fstream>
#include <cstdint>
#include <vector>
int main(int argc,char**argv){FileReader vcd(argv[2]);FileReader inter(argv[1],vcd.getFileEnd());interParser parser(nullptr);parser.parseHeader(inter.getPtr());vcdParser waves(parser.getWireMgr(),300000000);waves.parseAll(vcd.getPtr());std::ofstream f(argv[3],std::ios::binary);uint64_t n=parser.getTransC();f.write((char*)&n,8);for(uint64_t wire=4;wire<n+4;wire++){tHistory*h=(*parser.getWireMgr())[wire];uint64_t count=h->size();f.write((char*)&wire,8);f.write((char*)&count,8);std::vector<uint64_t>a;for(size_t k=0;k<count;k++){a.push_back((*h)[k].t);a.push_back((unsigned char)(*h)[k].v);}f.write((char*)a.data(),a.size()*8);}return 0;}
