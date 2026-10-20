#!/usr/bin/env python3
import argparse,json,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--mode',type=int,choices=range(3),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;c=json.loads((base/f'study_config_mode{a.mode}.json').read_text());old=str(Path(c['models']['cuda']['cwd']).parents[1]);new=str(a.source_root.resolve())
def relocate(v):
 if isinstance(v,str):return v.replace(old,new)
 if isinstance(v,list):return [relocate(x) for x in v]
 if isinstance(v,dict):return {k:relocate(x) for k,x in v.items()}
 return v
c=relocate(c);c={k:v for k,v in c.items() if not k.startswith('reuse_')};c['configuration_id']+='__cold_'+uuid.uuid4().hex[:12];c['cold_reproduction_note']='freshall4GPUseries; rawregionfieldaliasretained, separateimmutabletimingmetadatareviewneeded; noaggregatedacceptanceclaimfromthisunexecutedconfighelper';a.output.write_text(json.dumps(c,indent=2)+'\n');print(a.output)
