/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef KECCAKTREEGPU_H_INCLUDED
#define KECCAKTREEGPU_H_INCLUDED

#include "KeccakTree.h"
#include "KeccakTypes.h"
#include <iris/iris.hpp>

//************************
//First Tree mode
//data to be hashed is in h_inBuffer
//output chaining values hashes are copied to h_outBuffer
//************************

void KeccakTreeGPU(tKeccakLane * h_inBuffer, iris::DMem * d_inBuffer, 
                   tKeccakLane * h_outBuffer, iris::DMem * d_outBuffer, 
                   iris::DMem * d_KeccakF_RoundConstants);


//error function
void  checkCUDAError(const char *msg);

#endif // KECCAKTREEGPU_H_INCLUDED
