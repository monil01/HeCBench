#define main legacy_bn_main
#include "native_reference_repaired_main.cu"
#undef main
int main(int argc,char**argv){int code=legacy_bn_main(argc,argv);int self=0;for(int g=0;g<HIGHEST;g++)for(int i=0;i<NODE_N;i++)self+=bestGraph[g][i][i];printf("CORRECTED_TOP3_SELF_EDGES=%d\n",self);return code!=0||self!=0;}
