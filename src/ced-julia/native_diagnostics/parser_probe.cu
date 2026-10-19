// Calls the unchanged original parser only; no GPU kernels are launched.
#define main original_ced_main
#include "../../ced-cuda/main.cu"
#undef main
int main(int argc,char**argv){Params p(argc,argv);unsigned char *frames[1];int rows,cols,size;read_input(frames,rows,cols,size,p);free(frames[0]);return 0;}
