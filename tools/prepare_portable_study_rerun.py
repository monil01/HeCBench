#!/usr/bin/env python3
"""Relocate a recorded four-model configuration for an explicitly separate replication.
Does not run or benchmark anything, and never reuses another machine's samples.
"""
import argparse,datetime,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True,help='GPU index or comma-separated indices for an explicitly scoped multi-GPU replication');p.add_argument('--original-root',default='/home/imo/HeCBench_julia');a=p.parse_args()
root=Path(__file__).resolve().parents[1];original=json.loads(a.config.read_text())
def relocate(value):
 if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
 if isinstance(value,list):return [relocate(v) for v in value]
 if isinstance(value,str) and value.startswith(a.original_root+'/'):return str(root)+value[len(a.original_root):]
 return value
c=relocate(original)
if set(c.get('models',{}))!={'cuda','cuda_jl','ka','jacc'}:raise SystemExit('Replication requires all four configured models.')
for k in ['reuse_result','reuse_models','allow_cross_configuration_reuse','cross_configuration_reuse_reason','reuse_reviewed_dependency_changes','reuse_legacy_adapter_unchanged_verified']:c.pop(k,None)
gpu_ids=a.gpu.split(',')
if not gpu_ids or any(not x.isdecimal() for x in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):raise SystemExit('Use distinct numeric GPU indices, e.g.6 or5,6.')
for spec in c['models'].values():spec.pop('reuse_run',None)
c['gpu']=int(gpu_ids[0]) if len(gpu_ids)==1 else ','.join(gpu_ids); c['replication_of_configuration']=original['configuration_id'];c['configuration_id']=original['configuration_id']+'_replication_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');c['replication_note']='Fresh four-model series; original frozen paper data unchanged. Use source revision matching the recorded primary configuration.'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(c,indent=2)+'\n');print(a.output)
