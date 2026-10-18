#include <cstdio>
#include <cstring>
#define fp float
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/file.c"
int main(){float p[18];for(auto&v:p)v=-999;read("/home/imo/HeCBench_julia/src/myocyte-julia/study_params.txt",p,18,1,0);for(int i=0;i<18;i++){unsigned bits;memcpy(&bits,&p[i],4);printf("%d %.9g %08x\n",i,p[i],bits);}}
