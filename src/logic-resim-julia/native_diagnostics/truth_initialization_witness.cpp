// Minimal actual CPU execution of the original packed-table |= statements.
// Controlled initial storage=0xff; not a GPU fault or the original default run.
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
int main(int argc,char**argv){std::ifstream in(argv[1]);std::string line;getline(in,line);std::istringstream header(line);size_t trans,iw,ow,assign,eval,blocks,start;header>>trans>>iw>>ow>>assign>>eval>>blocks>>start;size_t wrong=0,used=0;std::vector<unsigned char>poison(eval*1025,255),expected(eval*1025,0);for(size_t serial=0;serial<eval;serial++){getline(in,line);std::istringstream row(line);unsigned degree;std::string truth;row>>degree>>truth;size_t head=serial*1025,counter=0;for(char c:truth){poison[head+(counter>>2)]|=((c-'0')<<(counter&3)*2);expected[head+(counter>>2)]|=((c-'0')<<(counter&3)*2);counter++;}for(size_t k=0;k<truth.size()/4;k++){used++;wrong+=poison[head+k]!=expected[head+k];}}printf("controlled_CPU_original_bitpacking used_bytes=%zu wrong_from_poison=%zu observed_original_GPU_wrongwave=false\n",used,wrong);return wrong?0:1;}
