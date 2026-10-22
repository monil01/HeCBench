#!/usr/bin/env python3
"""Future root-only one-shot calibration. This source has not been imported/run."""
from pathlib import Path
import argparse,csv,fcntl,hashlib,json,math,os,re,shutil,subprocess
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(value,message):
 if not value:raise RuntimeError(message)
def dump(path,value):
 Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def locked_bindings(spec):
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH)
  changed=[name for name,digest in spec['bindings'].items() if not Path(name).is_file() or sha(name)!=digest]
  need(not changed,'Frozen scientific/source/method binding changed: '+repr(changed))

def audit(spec,root,collector_code):
 slot=root/'triad/cuda';provenance=json.loads((slot/'profile_provenance.json').read_bytes());profile=provenance.get('profile') or {};proof=profile.get('process_generation_ownership') or {}
 checks={'collector_exit_zero':collector_code==0,'status_completed':provenance.get('status')=='profile_process_completed','profile_exit_zero':profile.get('returncode')==0,'no_stop':profile.get('stop_reason') is None,'quiescent':profile.get('own_session_quiescent') is True,'no_unresolved_GPU':profile.get('unresolved_GPU_ownership')==[],'no_external_GPU':profile.get('external_gpu_activity')==[],'no_quiescence_unknowns':profile.get('own_session_quiescence_unknowns')==[],'no_safety_hold':profile.get('ownership_quiescence_hold') is None,'candidate_exact':provenance.get('collector_sha256')==spec['candidate_sha256'],'UUID_exact':provenance.get('gpu_uuid')==spec['gpu_uuid'],'method_exact':(provenance.get('profiler_method') or {}).get('review_sha256')==spec['method_review_sha256'],'expected_100_kernels':provenance.get('kernel_count')==100 and provenance.get('selection_expected_launch_count_matches') is True,'source_recheck_unchanged':provenance.get('binding_recheck_after')=='unchanged'}
 log=(slot/'execution.log').read_text();rows=[json.loads(line[len('STUDY_TRIAL '):]) for line in log.splitlines() if line.startswith('STUDY_TRIAL ')]
 checks['full_five_numerical_checks']=([r.get('trial') for r in rows]==[1,2,3,4,5] and all(r.get('correct') is True and r.get('checked_values')==4194304 and r.get('internal_repeats')==100 and r.get('timing_region')=='kernel_launch_loop_per_internal_repeat' and r.get('jit_excluded') is True and isinstance(r.get('max_abs_error'),(int,float)) and math.isfinite(r['max_abs_error']) and r['max_abs_error']<=1e-3 and r.get('output_artifact',{}).get('path')==str(slot/'scratch/profile_outputs'/('triad_'+str(r['trial'])+'.bin')) and r.get('output_artifact',{}).get('count')==4194304 and r.get('output_artifact',{}).get('dtype')=='float32' for r in rows))
 manifest=json.loads((slot/'profiling_auxiliary_manifest.json').read_bytes());files={x['path']:x for x in manifest}
 checks['full_five_output_manifest']=all(files.get('profile_outputs/triad_'+str(i)+'.bin',{}).get('bytes')==16777216 and re.fullmatch('[a-f0-9]{64}',files.get('profile_outputs/triad_'+str(i)+'.bin',{}).get('sha256','')) for i in range(1,6))
 checks['manifest_join']=sha(slot/'profiling_auxiliary_manifest.json')==provenance.get('auxiliary_manifest_sha256')
 checks['default_new_scratch_cleanup']=not Path(spec['raw_root'],'triad/cuda/scratch').exists()
 gens=proof.get('generations',[]);leader=proof.get('leader');events=proof.get('GPU_observations',[])
 checks['bootstrap_live_root_recorded']=any(g.get('identity',{}).get('pid')==leader and g.get('proof',{}).get('kind')=='fresh_Popen_start_new_session_anchor' and g.get('identity',{}).get('start_ticks') for g in gens)
 checks['all_cached_generations_dead_after_teardown']=bool(gens) and all(g.get('pidfd_live_at_snapshot') is False for g in gens)
 raw=(slot/'counters.csv').read_text();connected={int(p) for p in re.findall(r'^==PROF== Connected to process ([0-9]+) \('+re.escape(str(slot/'scratch/study_native'))+r'\)$',raw,re.M)}
 parsed=list(csv.reader(line for line in raw.splitlines() if line.startswith('"')))
 header=next((r for r in parsed if 'Process ID' in r and 'Metric Name' in r),None)
 counterpids={int(r[header.index('Process ID')]) for r in parsed if header and len(r)==len(header) and r[0].isdigit()}
 owned={e.get('pid') for e in events if (e.get('ownership') or {}).get('class')=='live_generation_owned' and e.get('gpu_uuid')==spec['gpu_uuid']}
 pinned={g.get('identity',{}).get('pid') for g in gens}
 checks['actual_target_live_ownership_observed']=len(connected)==1 and connected==counterpids and bool(connected&owned&pinned) and leader not in connected
 checks['no_retained_scan_unknowns']=proof.get('unresolved_session_candidate_pids')==[]
 # A historical unresolved scan/refresh interval cannot become a calibration pass.
 checks['no_unresolved_scan_history']=not any(e.get('kind')=='refresh_identity_unresolved' or (e.get('kind')=='strict_monitored_session_scan' and e.get('scan',{}).get('unresolved')) for e in events)
 return {'status':'single_triad_live_ownership_calibration_pass_no_deployment_authority' if all(checks.values()) else 'calibration_hold_or_failure_preserved_no_admission','checks':checks,'trial_rows':rows,'target_counter_PIDs':sorted(counterpids),'scope':'One accepted native triad fixture, live ownership/bootstrap/teardown/quiescence only. No foreign/PID-reuse/race completeness or TGVNN completion claim. Primary timing unchanged; no counter/capture auto-adoption.'}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--root-approval',type=Path);ap.add_argument('--independent-approval',type=Path);a=ap.parse_args()
 need(a.execute and a.root_approval and a.independent_approval,'Execution requires root and independent exact-packet approvals; no implicit check/run mode')
 spec=json.loads((P/'launch_spec.json').read_bytes());delivery_sha=sha(P/'DELIVERY.json');self_sha=sha(__file__)
 for path,role in [(a.root_approval,'root'),(a.independent_approval,'independent')]:
  approval=json.loads(path.read_bytes())
  need(approval.get('role')==role and approval.get('approved_for_single_076_launch') is True and approval.get('delivery_sha256')==delivery_sha and approval.get('controller_sha256')==self_sha and approval.get('candidate_sha256')==spec['candidate_sha256'],'Missing/mismatched exact-packet '+role+' approval')
 delivery=json.loads((P/'DELIVERY.json').read_bytes())
 for relative,item in delivery['files'].items():need(sha(P/relative)==item['sha256'],'Packet bytes changed: '+relative)
 with (P/'future_controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  root=Path(spec['raw_root']);authority=Path(spec['runtime_authority']);need(not root.exists() and not authority.exists(),'Fresh raw root and private authority required; no retry/reuse')
  locked_bindings(spec)
  need(shutil.disk_usage('/tmp').free>=spec['minimum_initial_free_bytes'],'Need 3GiB initial space, including 2GiB reserve; no cleanup performed')
  # Create only new directories after reviewed approval. The immutable template
  # ledger is copied as a distinct regular inode; source links are reads only.
  root.mkdir();slot=root/'triad/cuda';slot.mkdir(parents=True);scratch=slot/'scratch';scratch.mkdir()
  dump(root/'one_shot_attempt.json',{'collector_invocations_authorized':1,'retry_permitted':False,'packet_delivery_sha256':delivery_sha,'controller_sha256':self_sha,'candidate_sha256':spec['candidate_sha256'],'root_approval':{'path':str(a.root_approval),'sha256':sha(a.root_approval)},'independent_approval':{'path':str(a.independent_approval),'sha256':sha(a.independent_approval)}})
  shutil.copytree(spec['template'],authority)
  privateledger=authority/spec['ledger_relative'];canonical=Path(spec['canonical_ledger'])
  need(not privateledger.is_symlink() and privateledger.stat().st_ino!=canonical.stat().st_ino and sha(privateledger)==spec['canonical_ledger_sha256'],'Private ledger not an independent exact snapshot')
  (authority/'src').mkdir()
  for name,target in spec['source_symlinks'].items():Path(name).symlink_to(target,target_is_directory=True)
  for d in ['profile_outputs','ncu_tmp']:(scratch/d).mkdir()
  shutil.copyfile(P/'plan076.json',slot/'plan.json')
  for source in Path(spec['template']).rglob('*'):
   if source.is_file():need(sha(authority/source.relative_to(spec['template']))==sha(source),'Authority byte copy mismatch')
  need(sha(authority/'tools/v100_roofline_collect_ownership_proposal.py')==spec['candidate_sha256'],'Wrong alternate collector')
  locked_bindings(spec)
  env=os.environ.copy();env.update(spec['environment'])
  dump(root/'exact_launch.json',{'argv':spec['argv'],'environment_overrides':spec['environment'],'invocations':1,'process_budget_seconds':600,'unbounded_outside_deadline':['GPU-lock/idle waiting','fail-closed unknown-writer quiescence hold'],'canonical_ledger_writes':False})
  # No outer process timeout or signal: an unresolved writer must retain the
  # collector-held common GPU flock. Root must reconcile any crash/hold marker.
  with (root/'controller.log').open('xb') as log:result=subprocess.run(spec['argv'],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   locked_bindings(spec);report=audit(spec,root,result.returncode)
  except Exception as error:report={'status':'calibration_hold_or_failure_preserved_no_admission','error':repr(error),'collector_returncode':result.returncode}
  dump(root/'calibration_audit.json',report)
  print(json.dumps(report,allow_nan=False));return 0 if report['status']=='single_triad_live_ownership_calibration_pass_no_deployment_authority' else 1
if __name__=='__main__':raise SystemExit(main())
