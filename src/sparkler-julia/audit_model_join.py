#!/usr/bin/env python3
"""Exact per-call source/target capture joins plus rehash EVERY unique CAS byte.
Native's separately retained full arithmetic oracle audit is not rerun or inferred.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'.porting-state/portable_julia/sparkler_primary');p.add_argument('--model',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
def load(model):
 groups={};files=[]
 for f in sorted((a.root/model).glob('*_trial[1-5].jsonl')):
  nonce,rank,trial=f.stem.rsplit('_',2);key=(int(rank[4:]),int(trial[5:]));g=groups.setdefault(nonce,{})
  assert key not in g;g[key]=[json.loads(x)for x in f.read_text().splitlines()];files.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 assert len(groups)==1,groups.keys();g=next(iter(groups.values()));assert set(g)=={(r,t)for r in range(4)for t in range(1,6)};return g,files
ref,rf=load('cuda');target,tf=load(a.model);unique={};records=0
for key in ref:
 assert ref[key]==target[key],('complete capture ledger values/shape/hash differ',key)
 for r in target[key]:records+=1;unique[r['sha256']]=r.get('bytes',512)
bytes_=0
for sha,n in unique.items():
 raw=gzip.decompress((a.root/'CAS'/(sha+'.bin.gz')).read_bytes());assert len(raw)==n and hashlib.sha256(raw).hexdigest()==sha;bytes_+=len(raw)
oracle=Path(__file__).with_name('native_full_capture_audit.json');o=json.loads(oracle.read_text());assert o['complete_input_bit_oracle']and o['complete_C_lcm_integer_oracle']
r={'model':a.model,'reference':'cuda','all4ranks_all5trials':True,'exact_per_call_input_output_guard_metadata_and_actual_fullbyteSHA_joins':True,'ledger_rows_joined':records,'unique_CAS_frames':len(unique),'unique_CAS_bytes_rehashed':bytes_,'native_independent_full_input_C_oracle_receipt':str(oracle),'native_oracle_receipt_sha256':hashlib.sha256(oracle.read_bytes()).hexdigest(),'native_ledger_files':rf,'target_ledger_files':tf,'scope':'Complete fullactualbytes rehashed, everypercall source/target input/output/guard identityjoined; independent mathoracle is separatelyreviewednativeaudit+executedC++checker pertargetcall, notanother independentmathimplementation orclaim everyrepeatedBgenerationreadback.'}
a.receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if not k.endswith('files')}))
