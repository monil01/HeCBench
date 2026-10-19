#include <cstdio>
#include "../kmeans-cuda/kmeans.h"
int main(){float x0=0,x1=2;float*data[]={&x0,&x1};float c0=0,c1=.25;float*one[]={&c0};float*two[]={&c0,&c1};float r1=rms_err(data,1,2,one,1),r2=rms_err(data,1,2,two,2);int q1=r1,q2=r2;int selected=q2<q1?2:1;printf("RMSE_SELECTION_WITNESS original_rms_err k1=%.9g k2=%.9g int1=%d int2=%d source_selected=%d real_RMSE_prefers=2\n",r1,r2,q1,q2,selected);return !(r2<r1&&selected==1);}
