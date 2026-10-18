#define main original_main
#include "../atomicSystemWide-cuda/main.cu"
#undef main
extern "C" void study_cpu(int *a,int n,int loops){atomicKernel_CPU(a,n,loops);}
