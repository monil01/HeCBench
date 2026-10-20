#pragma once
#include <cstdint>
#include <cstddef>
struct NBLaunch{
 int32_t nAtoms,nCJ,nSCI,nEX,numTypes,gridZ,repeat,calcShift;
 float cutoffSq,beta,epsFac,padding;
 const void*pointers[9];
};
static_assert(sizeof(NBLaunch)==120 && offsetof(NBLaunch,pointers)==48,"NBLaunch ABI");
using NBCallback=int(*)(const NBLaunch*);
