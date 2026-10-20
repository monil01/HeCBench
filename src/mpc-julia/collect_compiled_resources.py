#!/usr/bin/env python3
"""Read-only producer-log resolution across immutable chained model reuse."""
from pathlib import Path
import json,re,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;S=ROOT/'.porting-state/portable_julia'
RIDS={'primary':'20261020T004745451491Z_mpc','partial_corrected':'20261020T004823417418Z_mpc'}
PAT=re.compile(r'STUDY_KERNEL_RESOURCE model=(\w+) kernel=(\w+) registers=(\d+) memory=\(local = (\d+), shared = (\d+), constant = (\d+)\)(?: dynamic_shared=(\d+))? active_blocks_per_SM=(\d+)')
def producer_log(rid,model):
 seen=set()
 while True:
  assert rid not in seen;seen.add(rid);q=S/'runs'/rid;log=q/f'{model}_run1.log'
  if log.is_file():return log
  rid=json.loads((q/'result.json').read_text())['models'][model]['reused_from_run_id']
def run():
 resource={}
 for role,rid in RIDS.items():
  resource[role]={}
  for m in ['cuda_jl','ka','jacc']:
   log=producer_log(rid,m);matches=[]
   for row in PAT.finditer(log.read_text()):
    model,kernel,regs,local,shared,const,dynamic,active=row.groups();matches.append({'kernel':kernel,'registers_per_thread':int(regs),'local_allocation_bytes_per_thread':int(local),'static_shared_bytes':int(shared),'dynamic_shared_bytes':int(dynamic or 0),'driver_max_active_blocks_per_SM_at1024threads':int(active)})
   assert len(matches)==2;resource[role][m]={'records':matches,'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
 report={'benchmark':'mpc','target_configuration_resources':resource,'native_reference_probe':{'source_log':'native_diagnostics/full_domain_run.log','scope':'SeparatestandaloneoriginalGPUbodydiagnosticNVCC12O3sm70; notper-primarybinaryattribute trace','registers':{'compress':29,'decompress':31},'shared_bytes_each':16392,'driver_max_active_blocks_each':2,'native_local_allocation_bytes':None},'definitions':'CUDA.memory(k).local is compiledper-threadlocalallocation, notisolatedspillbytesormeasuredtraffic. Driveractive_blocks is resource-derivedmaximum, notprofiledachievedoccupancyorformalforward-progressproof. Actualprimary/secondaryallfiveexecutions completed.','no_causal_claim':'No controlisolates templatepreload/layout/compilerwrapper/localmemorycontribution to timing; no counters/roofline/profiledtraffic.'};(P/'compiled_resources.json').write_text(json.dumps(report,indent=2)+'\n');print('Actualprimary/secondarycompiledresources collected withrecursiveproducerlogresolution')
if __name__=='__main__':run()
