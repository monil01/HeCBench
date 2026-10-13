#ifndef CU_H
#define CU_H
#if defined(__HIPCC__) || defined(__CUDACC__)
#define DEV __device__
#define HOST __host__
#define GLOBAL __device__
#else
#define DEV
#define HOST
#define GLOBAL
#endif
#endif

