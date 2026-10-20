#!/usr/bin/env python3
"""Owned fixed-fixture academic benchmark adapters; preserve original source."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;S=P.parent/'columnarSolver-cuda'
s=(S/'main.cu').read_text();defs=s[s.index('#define B '):s.index('#include "kernels.cu"')];(P/'source_defines.h').write_text(defs)
k=(S/'kernels.cu').read_text();k='''__device__ int bounded_key_index(float r){int i=(int)(r*KEY_LENGTH);return i<KEY_LENGTH?i:KEY_LENGTH-1;}
'''+k
k=k.replace('LCG_random_float(unsigned int * seed)','LCG_random_float(unsigned int * seed, unsigned int *draws)').replace('  const unsigned int m = 2147483648;','  (*draws)++;\n  const unsigned int m = 2147483648;',1)
k=k.replace('LCG_random_float(&localState)','LCG_random_float(&localState,&draws)');k=k.replace('LCG_random_float(&localState,&draws)*KEY_LENGTH','bounded_key_index(LCG_random_float(&localState,&draws))');assert k.count('bounded_key_index(')==6
k=k.replace('int *__restrict d_decrypted)','int *__restrict d_decrypted, unsigned int*diag)');k=k.replace('  unsigned int localState = globalState[idx];','  unsigned int localState = globalState[idx];\n  unsigned int draws=0;')
a='  for (int cycles=0; cycles<CLIMBINGS; ++cycles) {';assert a in k;k=k.replace(a,'  for(int q=0;q<KEY_LENGTH;q++)diag[idx*63+q]=(unsigned int)key[q];\n'+a)
a='  for (j=0; j<ENCRYPTEDLEN; ++j)\n    d_decrypted[idx*ENCRYPTEDLEN+j] = bestLocalDecrypted[j];';assert a in k;k=k.replace(a,a+'\n  for(int q=0;q<KEY_LENGTH;q++)diag[idx*63+30+q]=(unsigned int)key[q];\n  diag[idx*63+60]=__float_as_uint(bestScore);diag[idx*63+61]=localState;diag[idx*63+62]=draws;')
(P/'corrected_observed_kernels.cuh').write_text(k)
parser=s[s.index('bool extractBigrams'):s.index('bool verify')].replace('tempBigram[2]','tempBigram[3]').replace('"%s %f"','"%2s %f"');(P/'corrected_parser.inc').write_text(parser)
protocol={'benchmark':'columnarSolver','configuration_id':'columnarSolver_fixed397_KEY30_full150000x1024_parser_index_bounds_corrected_v1','predeclared_before_native_GPU':True,'parameters':{'key_length':30,'climbings_per_thread':150000,'blocks':32,'threads_per_block':32,'trajectories':1024,'encrypted_length':397,'bigram_entries':676},'scope':'Fixedsupplied397charplaintext/ciphertextanddeclaredoriginalbigramassetonly, no externaltargets/customcipherinputs','native_reference_repaired':True,'reference_provenance':{'original_source_hashes':{str(f.relative_to(P.parents[1])):hashlib.sha256(f.read_bytes()).hexdigest()for f in [S/'main.cu',S/'kernels.cu',S/'data']},'repairs':['char[3] and%2s parser fixes ASan3-bytewriteinto2-bytebuffer','Only5keyindexconversions clamp integer30to29; RNGvalues/draws/definedmoveformulasotherwiseunchanged'],'witnesses':['native_diagnostics/parser_asan.log','native_diagnostics/rng_domain_witness.json','native_diagnostics/cpu_bounds.log'],'original_GPU_failure_observed':False,'CPU_source_replay':'All1024original150000horizon:13keyindex30rejections/1011safefinishes; hostemulationevidence not originalGPUerror'},'input_contract':'ExactFloat32all676sourcebigramvalues,397Int32hardcodedcipherchars; actualseedstatesfromoriginalsetupKernelreadbackafterdecode(constinput); full1024outputs/initialandfinalpermutationkeys/finalbestscores/RNG/drawcounts/guards','acceptance':'Fullall1024candidatearrays/finalkey-score-RNGrecords vsCPUsource replay with independentlyvalidatedtransposition/bigrams/permutation/knownplaintextwinner. Defaultnativearithmetic inspected; no toleranceafterfailure. ExactIntoutputs, Float32scorebitsusedforsourcepathcomparison; independentFloat64score sum secondaryerrorreported.','GPU_body':'Source2kerneldefinitions retained;5key-indexclamps andtelemetryargs/stores/countersadded; mathematicalscopeconditionalcorrected, notoriginalbyte-verbatim','timing_region':'original_setupKernel_plus_full_decode_completed_whole_device','five_trials':'warmwholeworkload then5checkedtrials,mean2-5; H2D/reset/readbacks/oracle/serialization excludedall4','guard_scope':'256byteshead/tail all4sourceallocations+1diagnosticallocation; fullactualbuffers retained','storage_budget_new_bytes':250000000,'reserve_tmp_bytes':3000000000,'classification':'conditional_corrected_native_notexternalcryptanalysis_notframeworkunsupported'}
(P/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
# CPU source replay, bothfused/unfused builds, fixedfixtureonly.
prefix='''#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <omp.h>
#define __device__
#define __global__
#define __shared__
#define __restrict
#define __syncthreads() ((void)0)
struct dim {int x;};thread_local dim blockIdx,threadIdx,blockDim;
unsigned int __float_as_uint(float f){unsigned int u;memcpy(&u,&f,4);return u;}
#include "source_defines.h"
'''
post='''
void save(FILE*f,const void*p,size_t n){if(fwrite(p,1,n,f)!=n)abort();}
int main(int argc,char**argv){if(argc!=2)return 2;float scores[676]={};if(extractBigrams(scores,DATA_PATH))return 3;int encrypted[ENCRYPTEDLEN];for(int i=0;i<ENCRYPTEDLEN;i++)encrypted[i]=ENCRYPTED_T[i]-'a';unsigned int states[THREADS];int*out=new int[THREADS*ENCRYPTEDLEN];unsigned int*diag=new unsigned int[THREADS*63];
#pragma omp parallel for num_threads(16) schedule(dynamic,1)
for(int idx=0;idx<THREADS;idx++){blockIdx.x=idx/32;threadIdx.x=idx%32;blockDim.x=32;setupKernel(states);decode(scores,encrypted,states,out,diag);}
FILE*f=fopen(argv[1],"wb");save(f,states,sizeof(states));save(f,out,THREADS*ENCRYPTEDLEN*4);save(f,diag,THREADS*63*4);fclose(f);delete[]out;delete[]diag;return 0;}
'''
(P/'cpu_source_replay.cpp').write_text(prefix+parser+k.replace('  if (threadIdx.x == 0) {','  if(true) {')+post)
if __name__=='__main__':print(P/'protocol.json')
