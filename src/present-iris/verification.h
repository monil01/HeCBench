#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
static void verify_cipher(const unsigned char* actual,const unsigned char* reference,size_t bytes,int iteration){
 uint64_t hash=1469598103934665603ULL;size_t bad=0;
 for(size_t i=0;i<bytes;++i){if(actual[i]!=reference[i])++bad;hash^=actual[i];hash*=1099511628211ULL;}
 printf("CIPHER iteration=%d bytes=%zu hash=%016llx mismatches=%zu\n",iteration,bytes,(unsigned long long)hash,bad);
 if(bad)exit(2);
}
