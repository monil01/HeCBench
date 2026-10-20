// Source-specific cold recipe for the actual original uniform 1000/1000/100 case.
// Preserve the source mt19937_64 seed reset, uniform distribution and bit->i+1.
#include <random>
#include <vector>
#include <fstream>
#include <cassert>
#include <cstdint>
void u32(std::ofstream&f,uint32_t x){for(int i=0;i<4;i++)f.put(char((x>>(8*i))&255));}
void u64(std::ofstream&f,uint64_t x){for(int i=0;i<8;i++)f.put(char((x>>(8*i))&255));}
int main(int argc,char**argv){assert(argc==2);std::mt19937_64 rng(19937);std::uniform_int_distribution<> d(1,1000);std::vector<bool>bits(1000,false);for(int i=0;i<100;i++){int v=d(rng);assert(v>=0&&v<1000);bits[v]=true;}std::vector<unsigned>set;for(unsigned i=0;i<1000;i++)if(bits[i])set.push_back(i+1);std::ofstream f(argv[1],std::ios::binary);u32(f,1000);u32(f,1000);u64(f,uint64_t(set.size())*1000);for(int i=0;i<1000;i++)u32(f,set.size());for(int i=0;i<1000;i++)for(auto x:set)u32(f,x);return f?0:1;}
