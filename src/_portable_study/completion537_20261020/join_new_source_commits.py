#!/usr/bin/env python3
"""Join separately verified new-frontier source commits to effort metadata.
Original delivery JSON is preserved. No execution source, actual counters,
acceptance, raw rows, or timing samples are edited.
"""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--receipts',type=Path,required=True);a=p.parse_args()
receipt=json.loads(a.receipts.read_text());phases={x['phase']:x['commit']for x in receipt['commits']if x.get('returncode',0)==0};joins=[]
for model,phase,suffix in [('cuda_jl','shared','cuda.jl'),('ka','KA','KA'),('jacc','JACC','JACC')]:
 commit=phases[phase];main=R/f'src/{a.benchmark}-julia/{a.benchmark}-julia-{suffix}/main.jl';assert subprocess.check_output(['git','show',f'{commit}:{main.relative_to(R)}'],cwd=R)==main.read_bytes()
 main_sha=hashlib.sha256(main.read_bytes()).hexdigest()
 for f in [R/f'.porting-state/porting_logs/{a.benchmark}-julia-{model}/manifest.json',R/f'.porting-state/portable_julia/status_300/{a.benchmark}/{model}.json',main.parent/'translation_accounting.json',main.parent/'study_manifest.json']:
  if not f.exists():continue
  d=json.loads(f.read_text())
  if d.get('source_commit')==commit and d.get('coordinator_source_commit_join',{}).get('source_sha256')==main_sha:continue
  original_sha=hashlib.sha256(f.read_bytes()).hexdigest();history=R/f'src/{a.benchmark}-julia/reference_history/accounting_original_delivery_20261020/{model}_{f.name}_{original_sha[:12]}.json';history.parent.mkdir(parents=True,exist_ok=True)
  if not history.exists():history.write_bytes(f.read_bytes())
  d.update(source_commit=commit,commit_status='committed');d['coordinator_source_commit_join']={'commit':commit,'source':str(main.relative_to(R)),'source_sha256':main_sha,'original_metadata':str(history.relative_to(R)),'original_metadata_sha256':original_sha,'review':'git-show verified main bytes unchanged; metadata only, counters and GPU evidence unchanged','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};temp=f.with_suffix('.sourcejoin.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(f)
  joins.append({'path':str(f.relative_to(R)),'original_sha256':original_sha,'new_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'commit':commit})
receipt['metadata_joins']=joins;receipt['metadata_join_scope']='Ready packet original metadata hashes refer to archived pre-join documents; executed code/artifact bindings remain unchanged.';a.receipts.write_text(json.dumps(receipt,indent=2)+'\n');print(len(joins),'metadata source-commit joins')
