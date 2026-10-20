/* CPU-only status witness: no CUDA, no factorization, no GPU claim. */
#include <stdio.h>
#include <stdlib.h>
#include "nicslu.h"
#include "nicslu_util.h"
int main(int argc,char**argv){for(int k=1;k<argc;k++){uint__t n=0,z=0,*r=NULL,*p=NULL;real__t*a=NULL;SNicsLU s;NicsLU_Initialize(&s);int read=NicsLU_ReadTripletColumnToSparse(argv[k],&n,&z,&a,&r,&p);int mono=read==0;for(unsigned j=0;j<n&&mono;j++)mono=p[j]<=p[j+1];int create=read==0?NicsLU_CreateMatrix(&s,n,z,a,r,p):-999;int analyze=create==0?NicsLU_Analyze(&s):-999;printf("path=%s read=%d n=%u nnz=%u ptr_monotone=%d last=%u create=%d analyze=%d actual_n=%u\n",argv[k],read,n,z,mono,p?p[n]:0,create,analyze,s.n);NicsLU_Destroy(&s);free(a);free(r);free(p);}return 0;}
