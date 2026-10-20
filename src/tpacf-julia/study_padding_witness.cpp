// CPU-only original host bin setup and exact source waterfall bin selection.
#include <cstdio>
#define main original_tpacf_main
#include "original_native/main.cu"
#undef main
int source_bin(double dot,const double*b){for(int k=30;k>=0;k--)if(dot<b[k])return k+1;return 0;}
int main(){int nb=0;double*b=init_bins(5,1.f,10000.f,1,&nb);int selected=0;for(int i=0;i<NUMBINS-1;i++){if(0.0>b[i]){selected=i;break;}}int actual=source_bin(0.,b);printf("ZERO_BIN selected=%d actual_source_waterfall=%d nbins=%d\n",selected,actual,nb);int n=97176,pad=98304-n;printf("PADDING valid_points=%d padding=%d original_subtract_wrong_bin=%lld; all-pair count nonnegativity can fail\n",n,pad,1LL*n*pad);free(b);return selected==actual?2:0;}
