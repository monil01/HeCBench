// Exact allocation size and source tileComputeSymm accumulation loop, CPU-only.
#include <cstdlib>
#include <cstdio>
int main(){const int nBins=64,njk=10,nr=2;long long* RRs=(long long*)calloc(nBins*nr,sizeof(long long));unsigned int*subHistoTemp=(unsigned int*)calloc(njk*nBins,sizeof(unsigned int));subHistoTemp[0]=1;int file=nr-1;long long*histo=&RRs[nBins*file];for(int k=0;k<njk*nBins;k++)histo[k]+=subHistoTemp[k];printf("%lld\n",histo[0]);free(RRs);free(subHistoTemp);}
