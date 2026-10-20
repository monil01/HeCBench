#include <cstdio>
#include <cstddef>
#include "study_api.h"
int main(){printf("{\"sizeof_LCStage\":%zu,\"scalar_offset\":%zu,\"pointer_array_offset\":%zu,\"int_bytes\":%zu,\"pointer_bytes\":%zu}\n",sizeof(LCStage),offsetof(LCStage,scalar),offsetof(LCStage,p),sizeof(int),sizeof(void*));}
