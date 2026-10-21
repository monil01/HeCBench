#!/usr/bin/env python3
"""Recover Heartwall KA counter derivation from one exact completed saved run.

CPU-only: no subprocesses, CUDA APIs, builds, target runs or historical mutations.
Historical execution/source qualification comes exclusively from captured parent
provenance and snapshots; current source bytes never certify that execution.
"""
import argparse,copy,csv,datetime,hashlib,json,types
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source-root',type=Path,required=True)
 ap.add_argument('--raw-root',type=Path,required=True)
 ap.add_argument('--parent-slot',type=Path,required=True)
 ap.add_argument('--parent-provenance-sha256',required=True)
 ap.add_argument('--destination-batch',type=Path,required=True)
 a=ap.parse_args();root=a.source_root.resolve();raw=a.raw_root.resolve();parent=a.parent_slot.resolve();batch=a.destination_batch.resolve()
 if not raw.is_relative_to('/tmp')or raw==Path('/tmp')or not parent.is_relative_to(raw):raise SystemExit('Owned raw-root/parent required')
 if batch.exists()or not batch.is_relative_to(raw)or batch.parent!=raw or not batch.name.startswith('batch'):raise SystemExit('Require nonexistent fresh batch immediately under raw-root')
 pp=parent/'profile_provenance.json';original=pp.read_bytes()
 if sha(pp)!=a.parent_provenance_sha256:raise SystemExit('Exact parent provenance SHA mismatch')
 pr=json.loads(original);plan=json.loads((parent/'plan.json').read_text())
 if (plan.get('benchmark'),plan.get('model'))!=('heartwall','ka'):raise SystemExit('Only independently reviewed Heartwall KA parent is authorized')
 if pr.get('status')!='collection_failed' or pr.get('error')!='field larger than field limit (131072)':raise SystemExit('Parent failure is not the reviewed CSV field limit')
 process=pr.get('profile',{})
 if process.get('returncode')!=0 or process.get('stop_reason')is not None or process.get('external_gpu_activity')or not pr.get('finished_at'):raise SystemExit('Successful uncontended completed original target required')
 if pr.get('binding_recheck_after')!='unchanged' or not pr.get('additional_current_profile_input_bindings')or not pr.get('executed_target_path_bindings'):raise SystemExit('Captured parent execution/source binding checks required')
 if sha(parent/'plan.json')!=pr['plan_sha256'] or sha(parent/'counters.csv')!=pr['counters_sha256'] or sha(parent/'execution.log')!=process['log_sha256']:raise SystemExit('Parent captured plan/counter/log mismatch')
 if sha(parent/'collector_source.py')!=pr['collector_sha256']:raise SystemExit('Captured collector source SHA mismatch')
 parser_path=root/'tools/v100_roofline_classify.py';parser_bytes=parser_path.read_bytes();parser=types.ModuleType('recovery_parser');exec(compile(parser_bytes,str(parser_path),'exec'),parser.__dict__)
 old=csv.field_size_limit();csv.field_size_limit(64<<20)
 try:groups,counter_sha=parser.read_csv(parent/'counters.csv')
 finally:csv.field_size_limit(old)
 if not groups:raise SystemExit('No actual kernel rows recovered')
 expected=tuple(pr['expected_device_pci'][k]for k in ['domain','bus','device'])
 for g in groups:
  got=tuple(g['metrics'].get('device__attribute_pci_'+k+'_id')for k in ['domain','bus','device'])
  if got!=expected:raise SystemExit('Captured expected device PCI mismatch')
 calibration=root/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json';proof=json.loads(calibration.read_text())
 if proof['device_uuid']!=pr['gpu_uuid']or proof['architecture']!='sm_70':raise SystemExit('Bound half calibration device/architecture mismatch')
 half={'path':str(calibration.relative_to(root)),'sha256':sha(calibration),'half_lane_factor':2,'tensor_conversion_enabled':False,'reason':'Separately bound measured packed-half calibration; source/device identity retained from original execution, not current target source.'}
 ctx={k:plan[k]for k in ['benchmark','model','configuration_id','reference_result_sha256']}
 ctx.update(ROI_stage='unassigned',ROI_coverage='partial_first_function_geometry',first_invocation_may_be_warmup=True,cache_policy='all_cold_replay',clock_control='none',selector_scope=pr['selector_scope'],source_context_bound_before_after=True,gpu_sm70=True,fp16_work_semantics='Physical executed packed lanes, including unused scalarhalf high lanes; distinct from useful work.',counter_derivation_recovery=True,historical_source_identity='captured parent before/after qualification only')
 records=[parser.derive(g,half_lanes=2,half_provenance=half,context=ctx)for g in groups]
 if any(r['duration_seconds']is None or r['duration_seconds']<=0 or r['dram_bytes']is None for r in records):raise SystemExit('Recovered duration/DRAM unavailable')
 # No status/destination is created until the complete CPU parsing gates pass.
 slot=batch/'heartwall'/'ka';slot.mkdir(parents=True)
 copies={}
 for name in ['plan.json','counters.csv','execution.log','build.log','profiling_auxiliary_manifest.json','collector_source.py','write_guard_source.py']:
  p=parent/name
  if p.exists():data=p.read_bytes();q=slot/name;q.write_bytes(data);copies[name]={'source':bind(p),'copy':bind(q)}
 (slot/'parent_profile_provenance.json').write_bytes(original)
 if (parent/'parser_source.py').exists():(slot/'parent_parser_source.py').write_bytes((parent/'parser_source.py').read_bytes())
 (slot/'parser_source.py').write_bytes(parser_bytes)
 (slot/'recovery_tool_source.py').write_bytes(Path(__file__).read_bytes())
 dump(slot/'half_calibration_snapshot.json',proof)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 recovery={'schema_version':1,'recovery_kind':'saved_counter_CPU_derivation_only','parent':{'profile':str(pp),'sha256':a.parent_provenance_sha256,'status':pr['status'],'error':pr['error']},'raw_counter':bind(parent/'counters.csv'),'parent_plan':bind(parent/'plan.json'),'parent_execution_log':bind(parent/'execution.log'),'parent_collector':bind(parent/'collector_source.py'),'current_parser':bind(parser_path),'calibration':bind(calibration),'tool':bind(Path(__file__)),'copies':copies,'new_gpu_processes':0,'new_build_processes':0,'new_counter_samples':0,'reused_actual_counter_samples':len(groups),'field_size_limit_bytes':64<<20,'original_device_uuid':pr['gpu_uuid'],'original_expected_device_pci':pr['expected_device_pci'],'device_PCI_checks':'all saved rows match original captured UUID-to-PCI mapping','original_source_bindings':pr['additional_current_profile_input_bindings'],'original_executed_target_bindings':pr['executed_target_path_bindings'],'source_qualification':'Parent captured binding_recheck_after=unchanged. No today-source rehash used to certify historical execution.','original_source_or_device_requalification':False,'recovered_at':now}
 if pp.read_bytes()!=original or sha(parent/'counters.csv')!=counter_sha:raise SystemExit('Parent evidence changed during recovery')
 dump(slot/'counter_derivation_recovery.json',recovery)
 derived={'schema_version':1,'input_csv':str(slot/'counters.csv'),'input_sha256':counter_sha,'parser_sha256':hashlib.sha256(parser_bytes).hexdigest(),'kernel_count':len(records),'records':records,'benchmark_bottleneck':'not_aggregated','raw_metrics':groups,'counter_derivation_recovery':{'path':str(slot/'counter_derivation_recovery.json'),'sha256':sha(slot/'counter_derivation_recovery.json')}}
 dump(slot/'derived.json',derived)
 new=copy.deepcopy(pr);new.pop('error',None);new.pop('counter_parse_error',None)
 new.update(status='profile_process_completed',started_at=now,finished_at=now,plan=str(slot/'plan.json'),plan_sha256=sha(slot/'plan.json'),kernel_count=len(records),counters_sha256=counter_sha,operation_calibration=half,profile_device_identity_verified='Saved counter rows match parent captured UUID-to-PCI identity',counter_derivation_recovery_from={'path':str(pp),'sha256':a.parent_provenance_sha256,'receipt':str(slot/'counter_derivation_recovery.json'),'receipt_sha256':sha(slot/'counter_derivation_recovery.json')},CPU_only_recovery=True,new_gpu_processes=0,new_build_processes=0,original_execution_reused=True,original_execution_started_at=process.get('started_at'),original_execution_finished_at=process.get('finished_at'))
 dump(slot/'profile_provenance.json',new)
 print(json.dumps({'profile':str(slot/'profile_provenance.json'),'status':new['status'],'kernels':len(records),'new_gpu_processes':0,'new_build_processes':0}))
if __name__=='__main__':main()
