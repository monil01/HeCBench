#!/usr/bin/env python3
"""Independent exhaustive truth oracle for complete three-input source factor controls."""
import argparse,pathlib,json,struct,hashlib
p=argparse.ArgumentParser();p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--julia',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--diagnostic-layout',action='store_true');a=p.parse_args();raw=a.native.read_bytes();n=struct.unpack_from('<i',raw)[0];assert n==256;inputsize=4+(7*n+1)*4;expected=raw[inputsize:];got=a.julia.read_bytes()if a.julia else expected
raw_equal=got==expected
if a.julia and not a.diagnostic_layout:assert raw_equal
pos=0;table=struct.unpack_from('<'+'Q'*(4*n*8),got,pos);pos+=4*n*8*8;links=struct.unpack_from('<'+'i'*(4*n),got,pos);pos+=4*n*4;lens=struct.unpack_from('<'+'i'*(2*n),got,pos);pos+=2*n*4;nextrow=struct.unpack_from('<i',got,pos)[0];pos+=4;assert pos==len(got);assert 2*n<=nextrow<=4*n
variables=[sum(((j>>i)&1)<<j for j in range(8))for i in range(3)]
def canonical(output):
 tab=struct.unpack_from('<'+'Q'*(4*n*8),output);lnk=struct.unpack_from('<'+'i'*(4*n),output,4*n*8*8);lens0=struct.unpack_from('<'+'i'*(2*n),output,4*n*8*8+4*n*4);result=[]
 for row0,count in enumerate(lens0):
  row=row0;vals=[];seen=set()
  while len(vals)<count:
   assert row not in seen and 0<=row<4*n;seen.add(row);vals.extend(tab[row*8:row*8+min(8,count-len(vals))]);row=lnk[row]
  result.append(tuple(vals))
 return result
canonical_equal=canonical(got)==canonical(expected)
assert canonical_equal, 'Actual factor graphs differ beyond allocation row labels'
rows=[]
for index in range(n):
 for neg in range(2):
  row=2*index+neg;count=lens[row];assert 0<count<=253
  keys=[];seen=set();current=row
  while len(keys)<count:
   assert 0<=current<nextrow and current not in seen;seen.add(current);take=min(8,count-len(keys));keys.extend(table[current*8:current*8+take]);current=links[current]
   if len(keys)<count:assert current>0
  def decode(q):return q>>32,(q&0xffffffff)>>1,q&1
  l,r,c=decode(keys[0]);vals=variables.copy()
  if count==1 and l==r:
   assert c==(l&1);node=l>>1;assert 0<=node<=3;function=(255 if node==0 else variables[node-1])^(255 if l&1 else 0)
  else:
   for offset,q in enumerate(keys):
    l,r,c=decode(q);limit=3+offset;assert 0<=l//2<limit and 0<=r//2<limit
    vals.append((vals[l//2]^(255 if l&1 else 0))&(vals[r//2]^(255 if r&1 else 0)))
   function=vals[-1]^(255 if c else 0)
  assert function==index,('independent truth mismatch',index,neg,count,function)
  rows.append(dict(truth_function=index,negative_cover_branch=bool(neg),internal_nodes=count))
r=dict(scope='OriginalfactorFromTruthunit, all256threeinputfunctionsandbothcoverpolarities; notfullresyn2',native_sha256=hashlib.sha256(raw).hexdigest(),native_inputs_bytes=inputsize,julia_sha256=hashlib.sha256(got).hexdigest()if a.julia else None,raw_crossmodel_bytes_compared=len(got)if a.julia else 0,cross_model_comparison_performed=bool(a.julia),all_512_subgraph_truth_functions_exact=True,all_subgraph_edges_and_link_chains_valid=True,independent_boolean_comparisons=len(rows)*8,source_nextrow=nextrow,max_internal_nodes=max(x['internal_nodes']for x in rows),performance_measurement=False,raw_identity_gate_pass=raw_equal,canonical_all_ordered_subgraph_keys_equal=canonical_equal,diagnostic_only=bool(a.diagnostic_layout),primary_raw_gate_verdict_preserved='PASS'if raw_equal else'REJECTED',full_pipeline_verified=False)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
