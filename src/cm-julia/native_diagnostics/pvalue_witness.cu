#include "../../cm-cuda/process.cu"
int main(){for(int k: {0,1,49999,50000,50001,99999,100000})printf("source_computePValueHelper nAbove=%d nRandom=100000 result=%.17g\n",k,computePValueHelper(k,100000));}
