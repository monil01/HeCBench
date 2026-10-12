/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_HIP_DEFINITIONS_
#define CUHD_HIP_DEFINITIONS_

#include <iostream>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
struct uint4 {
    unsigned int x, y, z, w;
};
#endif

// CUERR may be called for error checking
#define CUERR {                                                              \
    hipError_t err;                                                         \
    if ((err = hipGetLastError()) != hipSuccess) {                         \
       std::cout << "HIP error: " << hipGetErrorString(err) << " : "       \
                 << __FILE__ << ", line " << __LINE__ << std::endl;          \
       exit(1);                                                              \
    }                                                                        \
}

#endif /* CUHD_HIP_DEFINITIONS */
