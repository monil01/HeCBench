#!/usr/bin/env python3
"""Exact CPU-oracle comparison of immutable saved Int64 output artifacts."""
from pathlib import Path
import argparse,json,hashlib
ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--mode',required=True,choices=['from_fp','to_fp']);ap.add_argument('--output');a=ap.parse_args();p=Path(__file__).resolve().parent;expected=p/f'expected_{a.mode}_i64.bin';want=expected.read_bytes();r=json.load(open(a.result));out={'benchmark':'intrinsics-cast','configuration_id':r['configuration_id'],'result':a.result,'method':'exact_all_bytes_cpu_oracle_not_float64_coerced_difference','expected_file':str(expected),'expected_sha256':hashlib.sha256(want).hexdigest(),'count':len(want)//8,'models':{}}
for model,z in r['models'].items():
 if not z.get('complete'):continue
 rows=[]
 for q in z['trials']:
  d=q['output_artifact'];f=Path(d['saved_path']);got=f.read_bytes();assert hashlib.sha256(got).hexdigest()==d['sha256'];assert d['dtype']=='int64'and d['count']*8==len(want);assert got==want,(model,q['trial']);rows.append({'trial':q['trial'],'checked_values':len(want)//8,'all_bytes_equal':True,'artifact':str(f),'sha256':d['sha256']})
 assert len(rows)==5;out['models'][model]=rows
assert {'cuda','cuda_jl'}<=out['models'].keys()
s=json.dumps(out,indent=2)+'\n'
if a.output:Path(a.output).write_text(s)
else:print(s)
