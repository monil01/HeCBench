#include <cstdio>
#include <cstring>
#include <cstddef>
#include "reference_history/rewrite.h"
int main(){rw::Cut c{};c.truthtable=0xffff;c.value=0x7ff;c.nLeaves=15;c.used=true;unsigned word=0;memcpy(&word,(char*)&c+20,4);printf("{\"Cut_size\":%zu,\"Cut_align\":%zu,\"Cut_used_byte\":%zu,\"packed_word\":%u,\"TableNode_size\":%zu,\"Library_size\":%zu}\n",sizeof(c),alignof(rw::Cut),offsetof(rw::Cut,used),word,sizeof(rw::TableNode),sizeof(rw::Library));}
