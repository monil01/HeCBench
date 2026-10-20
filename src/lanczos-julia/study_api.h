#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {int stage,n,nnz,group,blocks;double scalar;void*p[5];} LCStage;
typedef int (*LCcallback)(const LCStage*);
int study_call(const LCStage*);
void study_tridiag(const double*,const double*,int);
int lanczos_run(int,LCcallback,const char*);
#ifdef __cplusplus
}
#endif
