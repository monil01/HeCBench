#!/usr/bin/env python3
"""Full-byte native/Julia plus independent exhaustive reconstructed-AIG unit gate.
This is the one-candidate source Rewrite core unit, not the full resyn2 pipeline.
"""
import argparse,hashlib,json,pathlib,struct
p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--julia',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
raw=a.input.read_bytes();n,pi,k=struct.unpack_from('<iii',raw);off=12+4*k
original=[]
for _ in range(2):original.append(struct.unpack_from('<'+'i'*(n+1),raw,off));off+=4*(n+1)
x=a.native.read_bytes();y=a.julia.read_bytes();assert x==y
old,nn=struct.unpack_from('<ii',y);assert old==n;off=8;out=[]
for _ in range(2):out.append(struct.unpack_from('<'+'i'*(nn+1),y,off));off+=4*(nn+1)
replace=struct.unpack_from('<'+'i'*(n+1),y,off);off+=4*(n+1);assert off==len(y)
mask=(1<<(1<<pi))-1
def evaluate(left,right):
 vals=[mask]+[sum(((j>>i)&1)<<j for j in range(1<<pi)) for i in range(pi)]
 for node in range(pi+1,len(left)):
  l,r=left[node],right[node];assert 0<=l//2<node and 0<=r//2<node
  vals.append((vals[l//2]^(mask if l&1 else 0))&(vals[r//2]^(mask if r&1 else 0)))
 return vals
v=evaluate(*original);w=evaluate(*out);rows=[]
for node in range(n+1):
 assert v[node]==w[node],('old graph function changed',node)
 if replace[node]!=-1:
  q=replace[node];assert 0<=q//2<=nn
  assert v[node]==w[q//2]^(mask if q&1 else 0),('replacement function changed',node,q)
  rows.append(dict(original_node=node,replacement_literal=q,assignments_checked=1<<pi))
assert rows
result=dict(scope='One-candidate13 sourceRewrite GPU core plus original CPU reconstruction; no fullresyn2 claim',old_nodes=n,new_nodes=nn,primary_inputs=pi,all_outputs_byte_identical=True,raw_compared_bytes=len(x),all_reconstructed_nodes_topological=True,all_old_and_replacement_functions_exhaustively_preserved=True,boolean_comparisons=(n+1+len(rows))*(1<<pi),replacements=rows,inputs_sha256=hashlib.sha256(raw).hexdigest(),native_sha256=hashlib.sha256(x).hexdigest(),julia_sha256=hashlib.sha256(y).hexdigest(),performance_measurement=False)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
