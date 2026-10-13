#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
inline void deredundancy_dump(const int* lengths,const long* offsets,const long* words,
 const char* reads,const unsigned short* indices,const unsigned short* orders,
 const unsigned int* packed,const int* gaps,const int* magic,const int* cutoff,
 const int* clusters,const unsigned short* table,int n) {
 static_assert(sizeof(long)==8 && sizeof(int)==4 && sizeof(unsigned short)==2,"source ABI widths");
 const char* path=getenv("DEREDUNDANCY_DUMP"); if(!path) return;
 FILE* f=fopen(path,"wb"); if(!f) abort();
 auto put=[&](const void* p,size_t bytes){if(fwrite(p,1,bytes,f)!=bytes) abort();};
 put(&n,sizeof(n)); put(lengths,n*sizeof(int)); put(offsets,(n+1)*sizeof(long)); put(words,n*sizeof(long));
 put(gaps,n*sizeof(int)); put(magic,n*4*sizeof(int)); put(cutoff,n*sizeof(int)); put(clusters,n*sizeof(int)); put(table,65536*sizeof(unsigned short));
 for(int i=0;i<n;++i) {
   // Compare every defined element, never source-uninitialized alignment padding.
   put(reads+offsets[i],lengths[i]);
   put(packed+offsets[i]/16,((lengths[i]-gaps[i])/16+1)*sizeof(unsigned int));
   put(indices+offsets[i],lengths[i]*sizeof(unsigned short));
   put(orders+offsets[i],words[i]*sizeof(unsigned short));
 }
 if(fclose(f)) abort();
}
