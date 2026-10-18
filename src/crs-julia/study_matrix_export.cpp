#include <cstdio>
#include <cstdlib>
#include "../crs-cuda/GCRSMatrix.h"
int main(){for(int m=1;m<=4;m++)for(int w=4;w<=8;w++)for(int k=m;k<=4;k++){auto*b=gcrs_create_bitmatrix(k,m,w);auto*c=gcrs_create_column_coding_bitmatrix(k,m,w,b);printf("%d %d %d",k,m,w);for(int i=0;i<k*w;i++)printf(" %u",c[i]);puts("");free(b);free(c);}}
