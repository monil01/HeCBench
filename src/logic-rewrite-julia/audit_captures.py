#!/usr/bin/env python3
"""Independent complete netlist/truth audit of all retained checked trial captures."""
import argparse,hashlib,json,pathlib,struct
from aiger_controls import evaluate,read,PASSES
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();data=json.loads(a.result.read_text());parent=pathlib.Path(__file__).resolve().parent
# Runner schema differs for rows vs top-level references; locate captured artifact descriptors recursively.
def descriptors(x):
 if isinstance(x,dict):
  if 'output_artifact' in x and isinstance(x['output_artifact'],dict):yield x['output_artifact']
  for v in x.values():yield from descriptors(v)
 elif isinstance(x,list):
  for v in x:yield from descriptors(v)
def parse_graph(raw,offset):
 np,no,nn=struct.unpack_from('<iii',raw,offset);assert np in (4,6,9,12) and no==32 and 0<=nn<100000
 size=np+nn+1;offset+=12
 def take(n):
  nonlocal offset
  x=struct.unpack_from('<'+'i'*n,raw,offset);offset+=4*n;return x
 aa=take(size);bb=take(size);outs=take(no);refs=take(size);words=((1<<np)+63)//64;enc=raw[offset:offset+8*no*words];offset+=8*no*words
 mask=(1<<(1<<np))-1;vals=[mask]+[sum(((k>>j)&1)<<k for k in range(1<<np)) for j in range(np)];depths=[0]*(np+1)
 def literal(q):
  assert 0<=q<2*len(vals)
  return vals[q//2]^(mask if q&1 else 0)
 for j in range(np+1,size):
  assert 0<=aa[j]<2*j and 0<=bb[j]<2*j
  vals.append(literal(aa[j])&literal(bb[j]));depths.append(1+max(depths[aa[j]//2],depths[bb[j]//2]))
 truth=b''.join(literal(q).to_bytes(8*words,'little') for q in outs);assert enc==truth,'recorded truth differs from complete independent graph evaluation'
 oracle,expected=evaluate(parent/'controls'/f'control_pi{np}.aig');short=((1<<np)+7)//8
 padded=b''.join(expected[j*short:(j+1)*short].ljust(8*words,b'\0') for j in range(no));assert truth==padded,'original Boolean output mismatch'
 return dict(np=np,no=no,ands=nn,depth=max(depths[q//2] for q in outs),a=aa,b=bb,o=outs,refs=refs,truth_sha256=hashlib.sha256(truth).hexdigest()),offset
paths=[]
for d in descriptors(data):
 q=d.get('saved_path') or d.get('path');
 if not q:continue
 q=pathlib.Path(q)
 if not q.is_absolute():q=a.result.parent/q
 if q.exists() and q.suffix=='.bin' and q not in paths:paths.append(q)
# Receipts expose archived paths under different fields; explicit bundle discovery is safe,
# but do not infer success from discovery or omit failed/verdict metadata.
if not paths:paths=sorted(a.result.parent.glob('artifacts/**/*.bin'))
rows=[]
for q in paths:
 raw=q.read_bytes();offset=0;initial=[]
 for n in (4,6,9,12):
  g,offset=parse_graph(raw,offset);assert g['np']==n;initial.append(g)
 initial_end=offset;stages=[]
 for n,start in zip((4,6,9,12),initial):
  previous=start
  for k in range(15):
   inp,offset=parse_graph(raw,offset);out,offset=parse_graph(raw,offset);assert inp==previous,'stage input is not previous complete output';assert inp['np']==out['np']==n
   stages.append(dict(primary_inputs=n,pass_index=k+1,operation=PASSES[k],input_ands=inp['ands'],output_ands=out['ands'],output_depth=out['depth'],truth_sha256=out['truth_sha256']));previous=out
 assert offset==len(raw),'unparsed trailing capture bytes'
 rows.append(dict(path=str(q),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),input_prefix_int32_count=initial_end//4,checked_stages=len(stages),full_independent_boolean_pass=True,stages=stages))
assert rows,'No captured artifacts found'
result=dict(rows=rows,full_rehash_count=len(rows),unique_artifact_sha256=len({r['sha256'] for r in rows}),all_full_boolean_pass=True,scope='Complete netlist/topology/exhaustive truth per60stages; no physical/guard guarantee or256PI originalhyp equivalence inferred.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
