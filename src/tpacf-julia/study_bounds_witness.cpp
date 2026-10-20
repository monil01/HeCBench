// Host-only exact original init_bins plus exact upload byte count; no CUDA execution.
#include <cstring>
#define main original_tpacf_main
#include "original_native/main.cu"
#undef main
volatile unsigned char sink;
void writeBoundaries(double*p){unsigned char out[(NUMBINS-1)*sizeof(double)];memcpy(out,p,sizeof(out));for(auto b:out)sink=b;}
int main(){int nb=0;double*p=init_bins(5,1.f,10000.f,1,&nb);writeBoundaries(p);free(p);return 0;}
