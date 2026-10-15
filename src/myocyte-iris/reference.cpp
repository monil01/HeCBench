// Independent serial evaluation of every ECC/CAM equation, not a fallback.
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace std;
#include "define.h"
#define __device__
#include "kernel_ecc.cu"
#include "kernel_cam.cu"
int main(int argc,char**argv) {
 if(argc!=3)return 1;
 FILE*f=fopen(argv[1],"rb");if(!f)return 1;
 int t;fp in[EQUATIONS],p[PARAMETERS],out[EQUATIONS]={},com[3]={};
 if(fread(&t,sizeof(t),1,f)!=1||fread(in,sizeof(fp),EQUATIONS,f)!=EQUATIONS||fread(p,sizeof(fp),PARAMETERS,f)!=PARAMETERS)return 1;
 fclose(f);
 kernel_ecc(t,in,out,0,p);
 for(int k=0;k<3;++k)kernel_cam(t,in,out,46+15*k,p,5*k,com,k,in[35+k]*1e3);
 f=fopen(argv[2],"wb");if(!f)return 1;
 fwrite(out,sizeof(fp),EQUATIONS,f);fwrite(com,sizeof(fp),3,f);
 return fclose(f)!=0;
}
