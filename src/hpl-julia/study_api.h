#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef void (*hpl_gpu_callback)(int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
void hpl_observe(int,int,int,int,double,const double*,int,const double*,int,double,double*,int,const double*,const double*,const double*);
int hpl_initialize(void);
int hpl_run(int,hpl_gpu_callback,const char*);
void hpl_finalize(void);
#ifdef __cplusplus
}
#endif
