// CPU-only legal two-workgroup match output. No GPU execution claimed.
#include <algorithm>
#include <vector>
#include <cstring>
#include <iostream>
#include <cstdlib>
using uint=unsigned int;
#ifdef SAFE_VERIFY
#include "safe_verify.inc"
#else
#include "original_verify.inc"
#endif
int main(){const uint w=131072;std::vector<uint>r(2*w,0xffffffffU),gold;for(uint i=0;i<w-1;i++){r[i]=i;gold.push_back(i);}for(uint i=w;i<2*w;i++){r[i]=i;gold.push_back(i);}uint counts[]={w-1,w};return verify(counts,2,r.data(),w,gold);}
