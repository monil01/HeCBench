#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "original_native/args.h"
#define NUMBINS 64
void compileHistograms(long long* DDs, long long* DRs, long long* RRs, long long*** DD, long long*** DR, 
    long long*** RR, options *args) {
  *DD = (long long**)malloc(((*args).njk+1)*sizeof(long long*));
  *DR = (long long**)malloc(((*args).njk+1)*sizeof(long long*));
  *RR = (long long**)malloc(1*sizeof(long long*));
  for(int i=0; i<=(*args).njk; i++) {
    (*DD)[i] = (long long*)malloc(NUMBINS*sizeof(long long));
    (*DR)[i] = (long long*)malloc(NUMBINS*sizeof(long long));
    memset((*DD)[i], 0, NUMBINS*sizeof(long long));
    memset((*DR)[i], 0, NUMBINS*sizeof(long long));
  }
  (*RR)[0] = (long long*)malloc(NUMBINS*sizeof(long long));
  memset((*RR)[0], 0, NUMBINS*sizeof(long long));

  for(int i=0; i<NUMBINS; i++) {
    for(int k=0; k<(*args).njk; k++) {
      (*DD)[0][i] += DDs[NUMBINS*k + i];
    }
    for(int j=0; j<(*args).random_count; j++) {
      for(int k=0; k<(*args).njk; k++) {
        (*DR)[0][i] += DRs[(j*(*args).njk + k)*NUMBINS + i];
      }
      (*RR)[0][i] += RRs[j*NUMBINS + i];
    }
  }

  for(int k=1; k<=(*args).njk; k++) {
    for(int i=0; i<NUMBINS; i++) {
      (*DD)[k][i] = (*DD)[0][i] - DDs[(k-1)*NUMBINS + i];
      (*DR)[k][i] = (*DR)[0][i];
      for(int j=0; j<(*args).random_count; j++) {
        (*DR)[k][i] -= DRs[(j*(*args).njk + k - 1)*NUMBINS + i];
      }
    }
  }
}
int main(){long long d[128]={},r[64]={},c[128]={};d[0]=5;d[64]=1;options o{};o.njk=2;o.random_count=1;long long **DD,**DR,**RR;compileHistograms(d,c,r,&DD,&DR,&RR,&o);printf("SOURCE_DD full=%lld exclude_group0=%lld exclude_group1=%lld; exact_leave_one_group_out_for_2plus2_samebin_points=1,1\n",DD[0][0],DD[1][0],DD[2][0]);return DD[1][0]==1&&DD[2][0]==5?0:2;}
