#!/usr/bin/env python3
"""Independent full strash unit truth, topology and fanout-count gate."""
import argparse,pathlib,struct,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--julia',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
raw=a.input.read_bytes();n,pi,k=struct.unpack_from('<iii',raw);off=12+4*k;original=[]
for _ in range(2):original.append(struct.unpack_from('<'+'i'*(n+1),raw,off));off+=4*(n+1)
x=a.native.read_bytes();y=a.julia.read_bytes();assert x==y;nn,ppi,levels=struct.unpack_from('<iii',y);assert pi==ppi;off=12;out=[]
for _ in range(2):out.append(struct.unpack_from('<'+'i'*nn,y,off));off+=4*nn
outs=struct.unpack_from('<4i',y,off);off+=16;refs=struct.unpack_from('<'+'i'*nn,y,off);off+=4*nn;assert off==len(y)
mask=(1<<(1<<pi))-1
def evaluate(left,right):
 vals=[mask]+[sum(((j>>i)&1)<<j for j in range(1<<pi)) for i in range(pi)];depth=[0]*(pi+1)
 for node in range(pi+1,len(left)):
  l,r=left[node],right[node];assert 0<=l//2<node and 0<=r//2<node
  vals.append((vals[l//2]^(mask if l&1 else 0))&(vals[r//2]^(mask if r&1 else 0)));depth.append(1+max(depth[l//2],depth[r//2]))
 return vals,depth
v,_=evaluate(*original);w,depth=evaluate(*out)
for old,q in zip(range(n,n-4,-1),outs):assert 0<=q//2<nn and v[old]==w[q//2]^(mask if q&1 else 0)
expected=[0]*nn
for node in range(pi+1,nn):
 for lit in [out[0][node],out[1][node]]:expected[lit//2]+=1
for lit in outs:expected[lit//2]+=1
assert tuple(expected)==refs
r=dict(scope='FulloriginalbalancePerformV2operatorunit, notfullresyn2',raw_compared_bytes=len(x),all_bytes_identical=True,all_reconstructed_nodes_topological=True,all_fanout_counts_independently_exact=True,all_four_full_output_truth_tables_exact=True,exhaustive_boolean_comparisons=4*(1<<pi),input_AND_nodes=n-pi,output_AND_nodes=nn-pi-1,source_level_iterations=None,independent_output_depth=max(depth[q//2]for q in outs),input_sha256=hashlib.sha256(raw).hexdigest(),native_sha256=hashlib.sha256(x).hexdigest(),julia_sha256=hashlib.sha256(y).hexdigest(),PI_fanin_scope='UnitserializesonlyANDdevicefaninprefix; PIencodedzeroisexplicitnonsemanticpadding, notanactualdevicePIreadbackclaim',performance_measurement=False)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
