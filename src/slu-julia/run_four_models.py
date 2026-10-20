#!/usr/bin/env python3
"""Explicit fresh four-model campaign, checked native -> CUDA.jl -> KA -> JACC."""
from pathlib import Path
import argparse,json,subprocess,sys,os,uuid
P=Path(__file__).resolve().parent;R=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('--fixtures',type=Path,required=True);a.add_argument('--config-dir',type=Path,default=Path('/tmp')/('slu-four-model-configs-'+uuid.uuid4().hex));z=a.parse_args();z.config_dir.mkdir(parents=True,exist_ok=False);previous=None
for upto,model in enumerate(['cuda','cuda_jl','ka','jacc'],1):
 models=['cuda','cuda_jl','ka','jacc'][:upto];cfg=z.config_dir/(model+'.json');cmd=[sys.executable,str(P/'make_config.py'),'--models',','.join(models),'--fixtures',str(z.fixtures.resolve()),'--output',str(cfg),'--capture-root',str(z.config_dir.resolve()/'private_captures')]
 if previous:cmd+=['--reuse-result',str(previous)]
 subprocess.run(cmd,cwd=R,check=True)
 process=subprocess.run([sys.executable,str(P/'run_study.py'),'--config',str(cfg)],cwd=R,text=True,capture_output=True);print(process.stdout,end='');print(process.stderr,end='',file=sys.stderr)
 if process.returncode not in (0,2):raise RuntimeError('Central runner failed: '+str(process.returncode))
 receipts=[]
 for line in process.stdout.splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  if 'canonical_result_alias'in v:receipts.append(Path(v['canonical_result_alias']))
 if len(receipts)!=1:raise RuntimeError('Missing unique actual result receipt')
 previous=receipts[0];d=json.load(open(previous))
 for m in models:
  v=d['models'][m];assert v['complete']and v['provenance_valid']and v['status']=='native_check_pass'and len(v['trials'])==5 and all(t['correct']for t in v['trials'])
 if upto>=2:assert d['reference_agreement']
 if upto>=3:assert d['variant_verified']['ka']
 if upto==4:assert d['variant_verified']['jacc']and d['matched_performance_valid']
print(json.dumps({'all_four_verified':True,'final_result':str(previous),'warning':'A fresh campaign, not a replacement for archived measurements; keep all receipts and five-times.'}))
