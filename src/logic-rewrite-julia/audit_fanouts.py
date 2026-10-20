#!/usr/bin/env python3
"""Independent semantic fanout-count review of every complete captured AIG state."""
import argparse,pathlib,struct,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--result',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();x=json.loads(a.result.read_text());seen=set();rows=[]
def state(raw,off):
 np,no,nn=struct.unpack_from('<iii',raw,off);off+=12;n=np+nn+1;values=[]
 for count in (n,n,no,n):values.append(struct.unpack_from('<'+'i'*count,raw,off));off+=4*count
 aa,bb,outs,refs=values;off+=8*no*(((1<<np)+63)//64);expected=[0]*n
 for j in range(np+1,n):expected[aa[j]//2]+=1;expected[bb[j]//2]+=1
 for lit in outs:expected[lit//2]+=1
 differences=[dict(node=j,expected=want,observed=got)for j,(want,got)in enumerate(zip(expected,refs))if want!=got]
 return off,dict(primary_inputs=np,AND_nodes=nn,counted_nodes=n,mismatch_count=len(differences),first_mismatches=differences[:12])
for model in ('cuda','cuda_jl','ka','jacc'):
 if model not in x['models'] or not x['models'][model].get('complete'):continue
 for trial in x['models'][model]['trials']:
  q=pathlib.Path(trial['output_artifact']['saved_path'])
  if q in seen:continue
  seen.add(q);raw=q.read_bytes();off=0;states=[]
  for index in range(124):off,s=state(raw,off);s['state_index']=index;states.append(s)
  assert off==len(raw)
  rows.append(dict(model=model,path=str(q),sha256=hashlib.sha256(raw).hexdigest(),checked_states=124,total_counted_nodes=sum(s['counted_nodes']for s in states),mismatch_count=sum(s['mismatch_count']for s in states),mismatching_states=[s for s in states if s['mismatch_count']]))
r=dict(scope='Independentallincomingedge+POreferences for every node, all124completeAIGstates/trial',full_rehash_count=len(rows),all_full_fanout_counts_exact=all(row['mismatch_count']==0 for row in rows),total_counted_nodes=sum(row['total_counted_nodes']for row in rows),rows=rows)
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='rows'},indent=2))
