#!/usr/bin/env python3
"""Rehash and independently certify every saved full SA, input word and guard."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=json.loads(a.result.read_text());cfg=json.loads((a.result.parent/'config.json').read_text());cases=cfg['input_contract']['cases'];P=a.source_root/'src/sa-julia';rows=[];groups={};guard=0x13579bdf
models=result.get('models',{})
if isinstance(models,list):models={m['model']:m for m in models}
for model,m in models.items():
 for t in m.get('trials',[]):
  art=t.get('output_artifact',{});f=Path(art.get('saved_path',art['path']));raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==art['sha256'];v=np.frombuffer(raw,dtype='<i4');N=sum(c['n']+20 for c in cases);assert len(v)==2*N;off=0
  for c in cases:
   n=c['n'];size=n+20;inp=np.fromfile(P/c['input_file'],dtype='<i4');assert len(inp)==size;actual=v[off:off+size];out=v[N+off:N+off+size];assert np.array_equal(actual,inp);assert np.all(out[n:n+3]==0)and np.all(out[n+3:]==guard);sa=out[:n];assert np.all((sa>=0)&(sa<n));assert np.all(np.bincount(sa,minlength=n)==1);rank=np.empty(n+1,dtype=np.int32);rank[n]=-1;rank[sa]=np.arange(n,dtype=np.int32);x=inp[sa];tail=rank[sa+1];assert np.all((x[:-1]<x[1:])|((x[:-1]==x[1:])&(tail[:-1]<tail[1:])));off+=size
  trial=t.get('trial');groups.setdefault(trial,[]).append((model,h));rows.append({'model':model,'trial':trial,'path':str(f),'sha256':h,'bytes':len(raw),'all_input_words_exact':True,'all_suffix_positions_unique_lex_order_certified':True,'all_source_sentinels_external_guards_exact':True})
assert rows,'No accepted saved arrays found'
for trial,pairs in groups.items():assert len(set(h for m,h in pairs))==1,('different_complete_artifacts',trial,pairs)
out={'status':'pass','result':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((a.result.parent/'config.json').read_bytes()).hexdigest(),'artifact_count':len(rows),'artifact_total_bytes':sum(r['bytes']for r in rows),'models':sorted(set(r['model']for r in rows)),'certificate':'full permutation and strict(first symbol,inverse rank of tail) order; finite-tail descent establishes unique complete suffix order independently of DC3','all_models_complete_artifact_bytes_identical_per_trial':True,'NumPy_version':np.__version__,'scope':'final construction each outertrial only, no claim every internal R100 output or private workspace captured','rows':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}))
