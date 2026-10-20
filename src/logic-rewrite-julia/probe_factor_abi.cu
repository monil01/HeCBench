#include <cstdio>
#include <cstddef>
#include "reference_history/refactor.h"
#include "reference_history/sop.cuh"
#include "reference_history/vectors.cuh"
int main(){using S=sop::Sop;using M=VecsMem<unsigned,8192>;using G=subgUtil::Subg<256>;
 printf("{\"Sop\":{\"size\":%zu,\"align\":%zu,\"nLits\":%zu,\"nCubes\":%zu,\"pCubes\":%zu},\"VecsMem\":{\"size\":%zu,\"align\":%zu,\"nSize\":%zu,\"pArray\":%zu},\"Subg\":{\"size\":%zu,\"align\":%zu,\"nSize\":%zu,\"pArray\":%zu}}\n",sizeof(S),alignof(S),offsetof(S,nLits),offsetof(S,nCubes),offsetof(S,pCubes),sizeof(M),alignof(M),offsetof(M,nSize),offsetof(M,pArray),sizeof(G),alignof(G),offsetof(G,nSize),offsetof(G,pArray));}
