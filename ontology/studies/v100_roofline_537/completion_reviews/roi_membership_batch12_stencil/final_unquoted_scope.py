"""Omit unused original CLI text; retain full source hash. No target execution."""
import json
from pathlib import Path
O=Path(__file__).parent;p=O/'proposed_stage_patches.json';d=json.loads(p.read_text());excluded={}
for r in d['patches']:
 for sp in r['stage_patches']:
  pr=sp['updates']['ROI_membership_proof'];keep=[]
  for e in pr['evidence']:
   if e.get('start_byte',0):
    assert e['path']=='/home/imo/HeCBench_julia/src/myocyte-cuda/main.cu'
    assert pr['exact_stage_callsite']['path']!=e['path']
    excluded[e['path']]={'path':e['path'],'sha256':e['sha256'],'role':'Uncalled original standalone CLI main, not selected driver/kernel/callback; selected study_native.cu includes ECC/CaM/kernel only. Julia factory includes study_common and device code, not originalCLI.','reason':'NonUTF8 copyright prefix in unused file; keep exact full-byte binding, omit all textquote rather than skip or transcode header.','timed_stage_source_omitted':False}
   else:keep.append(e)
  pr['evidence']=keep;pr['timed_closure_boundaries']=[e for e in pr['timed_closure_boundaries']if e['path']not in excluded]
if excluded:
 d['unquoted_source_bindings']=list(excluded.values());p.write_text(json.dumps(d,indent=2)+'\n');(O/'unquoted_source_scope.json').write_text(json.dumps(list(excluded.values()),indent=2)+'\n')
