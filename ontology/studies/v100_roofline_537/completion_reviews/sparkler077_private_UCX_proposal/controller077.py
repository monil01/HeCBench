#!/usr/bin/env python3
"""Future one-shot root launch only; source preparation never imports/runs this."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess
P=Path(__file__).resolve().parent;MODELS=['cuda','cuda_jl','ka','jacc']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
 if not v:raise RuntimeError(m)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def canonical(s):
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);raw=Path(s['ledger']).read_bytes();l=json.loads(raw)
  need(l['accepted_named_contract']==json.loads(Path(s['accepted_contract_snapshot']).read_bytes()),'Accepted Sparkler scientific contract changed')
  changed=[p for p,h in s['bindings'].items() if not Path(p).is_file() or sha(p)!=h]
  need(not changed,'Frozen source/alias/plan/method/installedSDK changed: '+repr(changed))
  return l,hashlib.sha256(raw).hexdigest()
def qualified(m):return m.get('profiling_status')=='profile_process_completed' and m.get('sampled_kernel_count',0)>0

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--root-approval',type=Path);ap.add_argument('--independent-approval',type=Path);a=ap.parse_args();need(a.execute and a.root_approval and a.independent_approval,'Exact root and independent approval required before any runtime staging')
 s=json.loads((P/'launch_spec.json').read_bytes());dsha=sha(P/'DELIVERY.json');csha=sha(__file__)
 for path,role in [(a.root_approval,'root'),(a.independent_approval,'independent')]:
  receipt=json.loads(path.read_bytes());need(receipt.get('role')==role and receipt.get('approved_for_single_077_launch') is True and receipt.get('delivery_sha256')==dsha and receipt.get('controller_sha256')==csha,'Missing exact '+role+' packet approval')
 rootreview=json.loads(a.root_approval.read_bytes());need(rootreview.get('071_group_finished_and_supervisor_reaped') is True,'WAIT: root must certify actual071 ALL4 group finished and supervisor reaped')
 # Exact final provenance hashes supplied by root AFTER071completion, not
 # guessed from file existence while071 is active. This proposal binds no
 # incomplete071 future provenance SHA and never modifies or signals071.
 oldout={}
 for m in MODELS:
  path=Path(s['old071_root'])/'sparkler'/m/'profile_provenance.json';need(path.is_file(),'WAIT071: missing final '+m+' provenance')
  need(rootreview.get('final071_provenance_sha256',{}).get(m)==sha(path),'071final provenance not exactly root-reviewed: '+m)
  x=json.loads(path.read_bytes());need(bool(x.get('finished_at')) and x.get('status') not in ('started','waiting_for_idle','waiting_for_reviewed_artifact_space'),'071 not terminal: '+m);oldout[m]=x
 delivery=json.loads((P/'DELIVERY.json').read_bytes())
 for relative,item in delivery['files'].items():need(sha(P/relative)==item['sha256'],'Frozen packet changed: '+relative)
 with (P/'future_controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root=Path(s['new_root']);need(not root.exists(),'Fresh077root required; no retry/reuse')
  ledger,ledger_sha=canonical(s);active=[];skipped=[]
  for job in s['jobs']:
   m=job['model']
   if qualified(ledger['models'][m]):skipped.append(m);continue
   # Successful raw071 awaiting canonical qualification must be reviewed/adopted
   # or explicitly held by root; never replay it merely because adoption lags.
   need(not(oldout[m].get('status')=='profile_process_completed' and oldout[m].get('kernel_count',0)>0),'HOLD: successful raw071 '+m+' awaits canonical qualification, no duplicate077')
   active.append(job)
  if not active:
   print(json.dumps({'status':'all_variants_already_qualified_no_staging_no_invocation','skipped_models':skipped}));return 0
  need(shutil.disk_usage('/tmp').free>=s['required_free_bytes'],'Initial full5payload+3GiB margin+2GiBreserve gate not met; no cleanup or work reduction')
  root.mkdir();dump(root/'one_shot_attempt.json',{'collector_invocations_authorized':1,'retry':False,'delivery_sha256':dsha,'controller_sha256':csha,'071_final_reviewed':rootreview['final071_provenance_sha256'],'canonical_ledger_before_sha256':ledger_sha,'active_models':[j['model'] for j in active],'skipped_already_qualified_models':skipped,'root_approval_sha256':sha(a.root_approval),'independent_approval_sha256':sha(a.independent_approval)})
  for job in active:
   slot=root/'sparkler'/job['model'];scratch=slot/'scratch';scratch.mkdir(parents=True)
   for directory in ['profile_outputs','ncu_tmp','profile_source']:(scratch/directory).mkdir()
   for item in job['staged_files']:
    source=Path(item['source']);dest=Path(item['destination']);dest.relative_to(slot);need(sha(source)==item['sha256'],'Frozen source changed');need(not dest.exists(),'No existing destination overwrite');shutil.copyfile(source,dest);need(sha(dest)==item['sha256'],'Staged fulltext byte mismatch')
   shutil.copyfile(job['plan_source'],job['plan_runtime']);need(sha(job['plan_runtime'])==job['plan_sha256'],'Plan byte mismatch')
   plan=json.loads(Path(job['plan_runtime']).read_bytes())
   for name,h in plan['execution_bindings'].items():need(Path(name).is_file() and sha(name)==h,'Prospective execution binding failed: '+name)
  canonical(s)
  env=os.environ.copy();env.update(s['environment']);dump(root/'exact_launch.json',{'argv':s['argv'],'environment_overrides':s['environment'],'max_invocations':1,'process_budget_seconds':3600,'source_changes':'Only fresh execution prefixes +UCX_POSIX_DIR fresh private directory','no_UCC_UCX_transport_suppression':True,'default_new_scratch_full_manifest_then_cleanup':True})
  # No outer timeout/signals, retries, installations or shared source edits.
  with (root/'controller.log').open('xb') as log:result=subprocess.run(s['argv'],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT)
  reports=[]
  for job in active:
   slot=root/'sparkler'/job['model'];path=slot/'profile_provenance.json'
   if not path.is_file():reports.append({'model':job['model'],'status':'held_missing_provenance'});continue
   x=json.loads(path.read_bytes());profile=x.get('profile') or {};checks={'profile_completed':x.get('status')=='profile_process_completed','zero_return':profile.get('returncode')==0,'no_guard_stop':profile.get('stop_reason') is None,'source_rechecked':x.get('binding_recheck_after')=='unchanged','positive_sample_count':x.get('kernel_count',0)>0,'method_exact':(x.get('profiler_method') or {}).get('review_sha256')=='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca'}
   try:
    text=(slot/'execution.log').read_text();rows=[json.loads(l[len('STUDY_TRIAL '):]) for l in text.splitlines() if l.startswith('STUDY_TRIAL ')]
    checks['original_five_full_checks']=[r.get('trial') for r in rows]==[1,2,3,4,5] and all(r.get('correct') is True and r.get('checked_values')==40003517 and r.get('internal_repeats')==1 and r.get('timing_region')=='rank0_sum45_source_GEMM_stream_context_complete_MPI_barrier_intervals' and r.get('output_artifact',{}).get('count')==40003517 and r.get('output_artifact',{}).get('dtype')=='uint32' for r in rows)
    man=json.loads((slot/'profiling_auxiliary_manifest.json').read_bytes());files={r['path']:r for r in man};checks['full5_primary_manifest']=all(files.get('trial'+str(i)+'.bin',{}).get('bytes')==160014068 and re.fullmatch('[a-f0-9]{64}',files.get('trial'+str(i)+'.bin',{}).get('sha256','')) for i in range(1,6));checks['manifest_join']=sha(slot/'profiling_auxiliary_manifest.json')==x.get('auxiliary_manifest_sha256')
   except Exception as error:checks['saved_check_parse']=False;checks['saved_check_error']=repr(error)
   reports.append({'model':job['model'],'status':'raw_recovery_candidate_requires_root_scientific_admission' if all(v is True for v in checks.values()) else 'failure_or_hold_preserved_no_retry','checks':checks})
  dump(root/'finite_outcome.json',{'collector_returncode':result.returncode,'reports':reports,'skipped_models':skipped,'auto_retry':False,'no_fullROI_phase_or_causal_claim':True,'scope':'Source/path recovery outcome only. Root must independently audit all original MPI/vendor/CAS/hostchecks, rawcounter integrity, method and scientific admission. This controller does not validate or adopt a new profiler/ownership method.'})
  return result.returncode if result.returncode else (0 if all(r['status']=='raw_recovery_candidate_requires_root_scientific_admission' for r in reports) else 1)
if __name__=='__main__':raise SystemExit(main())
