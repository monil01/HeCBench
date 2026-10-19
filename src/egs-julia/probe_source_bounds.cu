// Original host source initializer; ASan diagnoses before any cudaMalloc.
#define CUDA_EGS
#include "configured_native/EGS.h"
#include "configured_native/media.c"
#include "reference_history/configured_init_before_spectrum_repair.cu"
int main(int argc,char**argv){if(argc!=2)return 2;spec_file=argv[1];init_source();}
