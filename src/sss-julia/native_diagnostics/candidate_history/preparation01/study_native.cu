#include "study_interface.h"
#include <cstdio>
#define myInt short
#define myBool bool
#define BLOCK_SIZE 32
#define SYNC __syncthreads()
#include "original_native/kernels.cu"
int native_callback(int op,int n,int shmem,void*stream,void*del,void*add,void*out,size_t,size_t){if(op==0)CanDeleteEdge<<<n,32,0,(cudaStream_t)stream>>>((short*)del,(short*)out);else CanAddEdge<<<n,32,shmem,(cudaStream_t)stream>>>((short*)del,(short*)add,(short*)out);return int(cudaGetLastError());}
int main(int argc,char**argv){if(argc!=4){fprintf(stderr,"usage: native DATA OUTPUT trials\n");return 2;}return sss_run(argv[1],argv[2],native_callback,atoi(argv[3]));}
