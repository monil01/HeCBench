#!/usr/bin/env python3
"""Independent full Boolean gate for every live source cut in the unit fixture."""
import argparse,pathlib,struct,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--native',type=pathlib.Path,required=True);p.add_argument('--julia',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();raw=a.native.read_bytes();n,np=struct.unpack_from('<ii',raw);off=8;arrays=[]
for k in range(3):arrays.append(struct.unpack_from('<'+'i'*(n+1),raw,off));off+=4*(n+1)
a0,b0,refs=arrays;cuts=raw[off:];assert len(cuts)==28*8*(n+1)
mask=(1<<(1<<np))-1;values=[mask]+[sum(((i>>j)&1)<<i for i in range(1<<np)) for j in range(np)]
def literal(q):return values[q//2]^(mask if q&1 else 0)
for j in range(np+1,n+1):assert a0[j]//2<j and b0[j]//2<j;values.append(literal(a0[j])&literal(b0[j]))
checked=0;live=0
for root in range(n+1):
 for slot in range(8):
  c=cuts[(root*8+slot)*28:(root*8+slot+1)*28]
  if not c[24]:continue
  sign,*leaves=struct.unpack_from('<5i',c);word=struct.unpack_from('<I',c,20)[0];count=(word>>27)&15;truth=word&65535
  assert 0<=count<=4
  live+=1
  for assign in range(1<<np):
   leaf_assign=sum(((values[leaves[j]]>>assign)&1)<<j for j in range(count))
   actual=(truth>>leaf_assign)&1;expected=(values[root]>>assign)&1
   assert actual==expected,(root,slot,assign,leaves,count,truth)
   checked+=1
result=dict(scope='source cut-enumeration subsystem unit, not full resyn2 benchmark',full_independent_cut_truth_pass=True,live_cuts=live,all_checked_boolean_values=checked,native_sha256=hashlib.sha256(raw).hexdigest(),native_source_original=True,refs_fixture='All14refcounts10; intentional unit-score input, not originalfullAIGfanoutcontract.')
if a.julia:
 got=a.julia.read_bytes();assert got==cuts
 result.update(julia_raw_full_byte_identity=True,julia_output_sha256=hashlib.sha256(got).hexdigest(),raw_cut_bytes=len(got))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
