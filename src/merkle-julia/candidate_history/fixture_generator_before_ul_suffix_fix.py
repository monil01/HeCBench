from pathlib import Path
import json,re,struct,random,hashlib,shutil
R=Path('/home/imo/HeCBench_julia');p=R/'src/merkle-julia';a=p/'merkle-julia-cuda.jl';a.mkdir(exist_ok=True)
for name in ['main.jl','Makefile']:
 if (p/name).exists()and not(a/('original_'+name)).exists():shutil.copy2(p/name,a/('original_'+name))
shutil.copy2(R/'src/merkle-cuda/LICENSE',p/'LICENSE')
P=2**64-2**32+1;hdr=(R/'src/merkle-cuda/rescue_prime.hpp').read_text();tables={}
for name in ['MDS','ARK1','ARK2']:
 body=re.search(r'inline constexpr ulong '+name+r'\[\d+\] = \{(.*?)\};',hdr,re.S).group(1);tables[name]=[int(x)for x in re.findall(r'(\d+)ull',body)]
assert [len(tables[x])for x in tables]==[144,84,84]
inverse=pow(7,-1,P-1);e3=4*(8+1);e4=e3*(64+1);e5=e4*(4096+1);e6=e5*64+e3;e7=e6*(2**31+1);exponent=4*(e6+2*e7)+7;assert exponent==inverse
mds=tables['MDS'];a1=tables['ARK1'];a2=tables['ARK2']
def perm(state):
 for rnd in range(7):
  state=[pow(v,7,P)for v in state];state=[(sum(mds[12*i+j]*state[j]for j in range(12))+a1[12*rnd+i])%P for i in range(12)]
  state=[pow(v,inverse,P)for v in state];state=[(sum(mds[12*i+j]*state[j]for j in range(12))+a2[12*rnd+i])%P for i in range(12)]
 return state

def tree(leaves,old=False):
 n=len(leaves)//4;out=[0]*(4*n);offset=n//2
 for i in range(offset):out[4*(offset+i):4*(offset+i+1)]=perm(leaves[(2 if old else 8)*i:(2 if old else 8)*i+8]+[0,0,0,8])[:4]
 offset//=2
 while offset:
  for i in range(offset):
   begin=8*offset+(2 if old else 8)*i;out[4*(offset+i):4*(offset+i+1)]=perm(out[begin:begin+8]+[0,0,0,8])[:4]
  offset//=2
 return out
rng=random.Random(919);cases=[]
for n in [64,128,256]:
 leaves=[rng.randrange(P)for _ in range(4*n)];leaves[:12]=[0,1,P-1,P-2,2**32-1,2**32,2**63,7,11,19,31,53];cases.append({'n':n,'leaves':leaves,'oracle':tree(leaves)})
original=tree(cases[0]['leaves'],True);changed=cases[0]['leaves'].copy();changed[-4:]=[(x+1)%P for x in changed[-4:]];original_changed=tree(changed,True);correct_changed=tree(changed)
assert original==original_changed;assert cases[0]['oracle'][4:8]!=correct_changed[4:8]
fixture=[len(cases)]+tables['MDS']+tables['ARK1']+tables['ARK2']
for c in cases:fixture +=[c['n']]+c['leaves']
(p/'fixtures_u64.bin').write_bytes(struct.pack('<'+'Q'*len(fixture),*fixture));oracle=[v for c in cases for v in c['oracle']];(p/'oracle_u64.bin').write_bytes(struct.pack('<'+'Q'*len(oracle),*oracle))
witness=cases[0]['leaves']+changed+original+original_changed+cases[0]['oracle']+correct_changed
(p/'original_witness_u64.bin').write_bytes(struct.pack('<'+'Q'*len(witness),*witness))
(p/'independent_oracle.json').write_text(json.dumps({'field_prime':P,'inverse7_exponent':inverse,'native_addition_chain_exponent':exponent,'method':'Python arbitraryprecision modularpow anddense12x12matrix products independently ofUInt64lazyreduction/UInt128highproduct/nativeadditionchain; alltreeinternalnodes fullvalueoracle','cases':[{'leaves':c['n'],'numeric_internal_field_words':4*(c['n']-1),'untouched_slot0':[0]*4,'root':c['oracle'][4:8]}for c in cases],'source_tree_stride_witness':{'original_stride_words':2,'required_nondisjoint_digestpair_stride':8,'lastleaf_perturbation_source_tree_unchanged':original==original_changed,'proper_tree_root_changes':cases[0]['oracle'][4:8]!=correct_changed[4:8],'source_root':original[4:8],'proper_root':cases[0]['oracle'][4:8],'proper_changed_root':correct_changed[4:8]},'scope':'Syntheticfinitepowers2 leaves; sourcehash constantsassumednot independentlycryptographicparametersecurity certification; no collisionresistance/realprotocolsecurity claim'},indent=2)+'\n')
# Minimal owned source fixes only two inputstrides, allremainingdeviceandhostbodies unchanged.
orig=(R/'src/merkle-cuda/merkle_tree.cu').read_text();assert orig.count('(DIGEST_SIZE >> 1)')==2;(p/'merkle_tree_corrected.inc').write_text(orig.replace('#include "merkle_tree.hpp"','#include "../merkle-cuda/merkle_tree.hpp"').replace('(DIGEST_SIZE >> 1)','(DIGEST_SIZE << 1)'))
print({'fixtures_words':len(fixture),'oracle_words':len(oracle),'witness_source_unchanged':original==original_changed})
