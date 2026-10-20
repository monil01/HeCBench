#!/usr/bin/env python3
"""CPU-only fixed original academic fixture bounds witness; no external inputs."""
from pathlib import Path
P=Path(__file__).resolve().parent;S=P.parents[1]/'columnarSolver-cuda'
s=(S/'main.cu').read_text();defs=s[s.index('#define B '):s.index('#include "kernels.cu"')]
k=(S/'kernels.cu').read_text();k=k.replace('  if (threadIdx.x == 0) {','  if (true) {')
k=k.replace('  if (posLeft != posRight)','  if(posLeft<0||posLeft>=KEY_LENGTH||posRight<0||posRight>=KEY_LENGTH)throw BoundFailure{CURRENT_IDX,CURRENT_CYCLE,posLeft,posRight,DRAWS,LAST_SEED};\n  if (posLeft != posRight)')
k=k.replace('  for (int cycles=0; cycles<CLIMBINGS; ++cycles) {','  for (int cycles=0; cycles<CLIMBINGS; ++cycles) { CURRENT_CYCLE=cycles;')
k=k.replace('  *seed = (a * (*seed) + c) % m;','  *seed = (a * (*seed) + c) % m; DRAWS++;LAST_SEED=*seed;',1)
prefix='''#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <omp.h>
#define __device__
#define __global__
#define __shared__
#define __restrict
#define __syncthreads() ((void)0)
struct dim {int x;}; thread_local dim blockIdx,blockDim,threadIdx;
thread_local int CURRENT_IDX=-1,CURRENT_CYCLE=-1;thread_local unsigned long long DRAWS=0;thread_local unsigned int LAST_SEED=0;
struct BoundFailure{int idx,cycle,left,right;unsigned long long draws;unsigned int state;};
'''
post='''
int main(){float scores[676]={};FILE*f=fopen(DATA_PATH,"r");char word[3];float value;int rows=0;while(fscanf(f,"%2s %f",word,&value)==2){scores[(word[0]-'a')*26+word[1]-'a']=value;rows++;}fclose(f);if(rows!=676)return 3;int encrypted[ENCRYPTEDLEN];for(int j=0;j<ENCRYPTEDLEN;j++)encrypted[j]=ENCRYPTED_T[j]-'a';unsigned int states[THREADS];int*out=new int[THREADS*ENCRYPTEDLEN];int fails=0,passes=0;
#pragma omp parallel for num_threads(16) schedule(dynamic,1)
for(int idx=0;idx<THREADS;idx++){blockIdx.x=idx/32;blockDim.x=32;threadIdx.x=idx%32;CURRENT_IDX=idx;CURRENT_CYCLE=-1;DRAWS=0;setupKernel(states);try{decode(scores,encrypted,states,out);__atomic_add_fetch(&passes,1,__ATOMIC_RELAXED);}catch(BoundFailure e){__atomic_add_fetch(&fails,1,__ATOMIC_RELAXED);
#pragma omp critical
{printf("BOUNDS_WITNESS {\\"thread\\":%d,\\"cycle\\":%d,\\"left\\":%d,\\"right\\":%d,\\"draws\\":%llu,\\"state\\":%u}\\n",e.idx,e.cycle,e.left,e.right,e.draws,e.state);fflush(stdout);}}
}
printf("CPU_FIXED_FULL_WORKLOAD {\\"threads\\":1024,\\"climbings\\":150000,\\"completed_safe\\":%d,\\"range_rejections\\":%d}\\n",passes,fails);delete[]out;return 0;}
'''
(P/'cpu_bounds_witness.cpp').write_text(prefix+defs+'\n'+k+'\n'+post)
