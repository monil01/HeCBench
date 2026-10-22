#!/usr/bin/env python3
"""Private finite SNICIT075 staging/controller proposal. Not executed.
Execution remains held until an independently reviewed ownership-monitor authority
is explicitly incorporated in a new packet; do not bypass this gate.
"""
from pathlib import Path
import argparse,json,hashlib,fcntl,os,shutil,subprocess
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/imo/HeCBench_julia')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def need(x,s):
 if not x:raise RuntimeError(s)
def load(p):return json.loads(Path(p).read_bytes())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def verify(pins):
 for p,h in pins.items():need(sha(p)==h,'Binding changed: '+p)
def current(j):
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);d=load(ROOT/'ontology/studies/v100_roofline_537/benchmarks/snicit.json')
 need(all(d['accepted_named_contract'][k]==v for k,v in j['expected_contract'].items()),'Accepted contract drift')
 m=d['models'][j['model']]
 if m['profiling_status']=='profile_process_completed'and m.get('sampled_kernel_count',0)>0:return 'qualified_skip'
 need(m['latest_profile_attempt']['profile_id']==j['expected_latest_profile_id'],'Latest attempt changed; new review required')
 return 'pending'
def stage(j):
 slot=Path(j['future_plan']).parent;s=Path(j['scratch']);need(not Path(j['future_root']).exists(),'Never overwrite/resume an attempt')
 slot.mkdir(parents=True,mode=0o700);s.mkdir(mode=0o700)
 for name in ['profile_outputs','profile_auxiliary','ncu_tmp']:(s/name).mkdir(mode=0o700)
 p=load(j['prepared_plan']['path'])
 for a in p['runtime_assets']:
  src=Path(a['source']);dst=Path(a['destination']);need(dst.is_relative_to(s),'Asset escapes fresh scratch');dst.parent.mkdir(parents=True,exist_ok=True)
  if src.is_dir():shutil.copytree(src,dst)
  else:shutil.copy2(src,dst)
  for e in a['source_files']:
   target=dst/Path(e['path']).relative_to(src)if src.is_dir()else dst;need(sha(target)==e['sha256'],'Copied runtime asset differs')
 for c in j['copies']:
  dst=Path(c['destination']);need(dst.is_relative_to(s),'Copy escapes scratch');need(sha(c['source'])==c['sha256'],'Private copy changed');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(c['source'],dst);need(sha(dst)==c['sha256'],'Copy differs')
 verify(p['execution_bindings']);Path(j['future_plan']).write_bytes(Path(j['prepared_plan']['path']).read_bytes());need(sha(j['future_plan'])==j['prepared_plan']['sha256'],'Staged plan differs')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();p=load(HERE/'proposal075.json');verify(p['tools']);need(len(p['models'])==4,'Exact four models required')
 rows=[]
 for j in p['models']:
  need(sha(j['prepared_plan']['path'])==j['prepared_plan']['sha256'],'Prepared plan changed');need(not Path(j['future_root']).exists(),'Fresh root already exists');rows.append({'model':j['model'],'status':current(j)})
 if not a.execute:print(json.dumps({'not_executed':True,'rows':rows,'execution_hold':'Pending independently reviewed ownership-monitor collector authority and new root-adopted packet; no staging, GPU, or build'}));return
 # Fail closed: current C21 provenance demonstrated unresolved child-ownership race.
 raise RuntimeError('075 EXECUTION HOLD: independently reviewed ownership monitor and root authorization are not incorporated; prepare a new frozen version, never bypass')
if __name__=='__main__':main()
