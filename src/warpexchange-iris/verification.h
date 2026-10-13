#pragma once
#include <cstdio>
#include <cstdint>
#include <cstddef>
static bool verify_mapping(const int* out,size_t n){
 uint64_t hash=1469598103934665603ULL;size_t mismatches=0;
 for(size_t i=0;i<n;++i){
   const size_t p=i%128, expected=(i/128)*128+(p%4)*32+p/4;
   // Original kernels leave out-of-range input registers undefined for partial logical warps.
   if(expected<n && out[i]!=static_cast<int>(expected)){
     if(mismatches<4)printf("Mapping mismatch %zu: actual %d expected %zu\n",i,out[i],expected);
     ++mismatches;
   }
 }
 const auto* bytes=reinterpret_cast<const unsigned char*>(out);
 for(size_t i=0;i<n*sizeof(int);++i){hash^=bytes[i];hash*=1099511628211ULL;}
 printf("OUTPUT_HASH %016llx count %zu\n",(unsigned long long)hash,n);
 if(n%128==0){printf("MAPPING %s mismatches %zu\n",mismatches?"FAIL":"PASS",mismatches);return mismatches==0;}
 printf("MAPPING partial warp: undefined source lanes excluded; not fully verified\n");return mismatches==0;
}
