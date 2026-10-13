#pragma once
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <type_traits>
template<class T> void summarize(const T* output,size_t n,const char* name){
  uint64_t hash=14695981039346656037ULL;size_t finite=0,infinite=0,nan=0;
  const unsigned char* bytes=reinterpret_cast<const unsigned char*>(output);
  for(size_t j=0;j<n*sizeof(T);++j){hash^=bytes[j];hash*=1099511628211ULL;}
  for(size_t i=0;i<n;++i){if(std::isnan(output[i]))++nan;else if(std::isinf(output[i]))++infinite;else++finite;}
  printf("CHECK type=%s kernel=%s n=%zu finite=%zu inf=%zu nan=%zu hash=%016llx\n",
      std::is_same<T,float>::value?"float":"double",name,n,finite,infinite,nan,(unsigned long long)hash);
}
