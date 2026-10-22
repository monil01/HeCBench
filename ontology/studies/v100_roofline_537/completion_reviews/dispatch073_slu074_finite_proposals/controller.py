#!/usr/bin/env python3
"""Private finite 073/074 proposal. Default is readonly validation. Root owns --execute.
No GPU query/signals; only exact C21 collector owns GPU6 idle/lock/guard/processes.
"""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,shutil,subprocess,fcntl
HERE=Path(__file__).resolve().parent

def need(x,msg):
 if not x:raise RuntimeError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
def save(p,x):
 t=p.with_suffix('.new');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def verify(bind):
 for p,h in bind.items():need(Path(p).is_file()and sha(p)==h,'Exact readonly binding absent/changed: '+p)
def generations(review):
 for e in review['predecessors']:
  p=Path('/proc')/str(e['pid'])
  try:
   f=(p/'stat').read_text().rsplit(')',1)[1].split()
   if f[19]==e['start_ticks']and f[0]not in ['Z','X']:raise RuntimeError('Original predecessor generation still live; no signals: '+str(e))
  except(FileNotFoundError,ProcessLookupError):pass

def current(job,root):
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);d=load(root/'ontology/studies/v100_roofline_537/benchmarks'/(job['benchmark']+'.json'))
 ct=d['accepted_named_contract'];s=d['models'][job['model']]
 need(all(ct[k]==v for k,v in job['expected_contract'].items()),'Selected scientific contract changed')
 if s['profiling_status']=='profile_process_completed'and s.get('sampled_kernel_count',0)>0:return 'qualified_skip'
 need(s['latest_profile_attempt']['profile_id']==job['expected_latest_profile_id'],'Latest failed attempt changed; new root review required')
 return 'pending'

def stage(job):
 dst=Path(job['future_plan_path']);slot=dst.parent;scratch=Path(job['scratch']);need(not slot.exists(),'Fresh attempt slot already exists; never resume/reconstruct')
 slot.mkdir(parents=True,mode=0o700);scratch.mkdir(mode=0o700)
 for name in ['profile_outputs','profile_auxiliary','ncu_tmp']:(scratch/name).mkdir(mode=0o700)
 plan=load(job['prepared_plan']['path'])
 for a in plan['runtime_assets']:
  source=Path(a['source']);dest=Path(a['destination']);need(dest.is_relative_to(scratch),'Runtime destination outside new scratch');dest.parent.mkdir(parents=True,exist_ok=True)
  if source.is_dir():shutil.copytree(source,dest)
  else:shutil.copy2(source,dest)
  for e in a['source_files']:
   sf=Path(e['path']);target=dest/sf.relative_to(source)if source.is_dir()else dest;need(sha(target)==e['sha256'],'Copied runtime asset mismatch')
 for cp in job['source_copies']:
  target=Path(cp['future_copy']);need(target.is_relative_to(scratch),'Copied wrapper outside owned scratch');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(cp['frozen_copy']['path']).read_bytes());need(sha(target)==cp['future_sha256'],'Frozen wrapper copy mismatch')
 for artifact in job['stage_artifacts']:
  target=Path(artifact['future_copy']);need(target.parent==slot and target.name=='build_manifest.json','Only reviewed slot compiler manifest stage allowed');need(not target.exists(),'Fresh compiler manifest already exists');target.write_bytes(Path(artifact['frozen_copy']['path']).read_bytes());need(sha(target)==artifact['future_sha256'],'Staged compiler manifest differs')
 for p,h in plan['execution_bindings'].items():need(sha(p)==h,'Staged execution binding differs')
 dst.write_bytes(Path(job['prepared_plan']['path']).read_bytes());need(sha(dst)==job['prepared_plan']['sha256'],'Staged plan differs')

def outcome(job):
 slot=Path(job['future_plan_path']).parent;p=slot/'profile_provenance.json'
 if not p.exists():return {'status':'no_provenance_inspect_log'}
 d=load(p);r={'status':d['status'],'profile':d.get('profile'),'provenance_sha256':sha(p),'auxiliary_manifest_sha256':d.get('auxiliary_manifest_sha256'),'binary_payloads_retained':False,'not_scientific_admission':True}
 if d['status']=='profile_process_completed':
  log=slot/'execution.log';text=log.read_text();rows=[json.loads(l[len('STUDY_TRIAL '):])for l in text.splitlines()if l.startswith('STUDY_TRIAL ')]
  need([x['trial']for x in rows]==[1,2,3,4,5],'Requires all five actual checks')
  need(all(x['correct']is True for x in rows),'Full scientific/execution checks incomplete')
  mp=slot/'profiling_auxiliary_manifest.json';need(sha(mp)==d['auxiliary_manifest_sha256'],'Full scratch manifest SHA mismatch');entries={e['path']:e for e in load(mp)}
  if job['benchmark']=='dispatch':
   need(all(x['checked_values']==0 and x['source_empty_kernel']is True and x['execution_checked']is True and x['numerical_agreement_kind']=='no_numerical_outputs_execution_only'and x['completed_source_loop_launch_calls']==1102200 and x['internal_repeats']==1000 and x['source_warmup_samples_per_mode']==100 and x['source_retained_samples_per_mode']==1000 for x in rows),'Original dispatch execution-only scope changed')
   r['execution_only_no_numerical_outputs']=True
  else:
   need('FullSLU warmup correct=1'in text,'Independent full warmup absent')
   need(text.count('SLU independent case=0 n=99340 pass=1')>=6,'Full ASIC warmup plus five independent checks absent')
   for row in rows:
    need(row['checked_values']==16784916 and row['internal_repeats']==1 and row['original_CPUpreprocess_symbolic_and_solve_execute_outside_ROI']is True,'Full SLU workload changed')
    need([c['case']for c in row['cases']]==[0,1,2,3],'Full four-case SLU suite absent')
    for c,n,nnz,terms,wsbytes in zip(row['cases'],[99340,4960,16,2048],[4008604,23942,32,4096],[557291235,43429,40,5120],[4113868080,205403520,662592,84811776]):
     need(c['n']==n and c['filled_nnz']==nnz and c['independent_LU_product_terms']==terms and c['independent_full_product_pass']is True and c['all_immutable_device_arrays_byte_exact']is True,'SLU independent case scope/check failed')
     captures=c['full_internal_workspace_captures'];need([v['role']for v in captures]==['full_actual_internal_workspace_before','full_actual_internal_workspace_after'],'Before/after full workspace absent')
     for v in captures:
      need(v['uncompressed_bytes']==wsbytes and v['all_actual_bytes_zero']is True and v['saved']is True and v['encoding']=='zlib_level1_lossless_actual_D2H_chunks','Full actual workspace check absent')
      e=entries[str(Path(v['path']).relative_to(job['scratch']))];need(e['bytes']>0 and len(e['sha256'])==64,'Full workspace artifact manifest absent')
    artifact=row['output_artifact'];p=Path(artifact['path']);rel=str(p.relative_to(job['scratch']))if p.is_absolute()else str(p);e=entries[rel]
    need(artifact['count']==16784916 and artifact['dtype']=='float64'and e['bytes']==134279328 and len(e['sha256'])==64,'Full SLU primary output absent')
  need(not Path(job['scratch']).exists(),'Canonical cleanup unexpectedly did not finish')
  r['all5_actual_checks_and_full_manifest_verified']=True
 return r

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--execute',action='store_true');a.add_argument('--review-sha256',required=True);a.add_argument('--group',choices=['dispatch','slu'],default='dispatch');args=a.parse_args()
 rp=HERE/'launch_review.json';need(sha(rp)==args.review_sha256,'Exact root-reviewed receipt SHA required');review=load(rp);need(sha(__file__)==review['controller_sha256'],'Controller drift');verify(review['tool_bindings']);verify(review['packet_bindings']);root=Path(review['source_root']);jobs=[j for j in review['jobs']if j['group']==args.group and not j['issues']];need(len(jobs)==(4 if args.group=='dispatch'else 1),'Exact finite job set differs')
 pending=[];skips=[]
 for j in jobs:
  verify(j['readonly_bindings']);need(sha(j['prepared_plan']['path'])==j['prepared_plan']['sha256'],'Prepared plan drift')
  for cp in j['source_copies']:need(sha(cp['frozen_copy']['path'])==cp['future_sha256'],'Frozen host wrapper drift')
  (skips if current(j,root)=='qualified_skip'else pending).append(j)
 generations(review)
 if not args.execute:print(json.dumps({'not_executed':True,'group':args.group,'pending':[j['job_id']for j in pending],'qualified_skips':[j['job_id']for j in skips],'held':[j['job_id']for j in review['jobs']if j['group']==args.group and j['issues']],'maximum_collector_invocations':1,'no_GPU_queries':True}));return
 # Execution is ROOT ONLY after exact source/method/path/storage review. No new
 # master, repeat round, historical restart, GPU signals or automatic retry.
 receipt=HERE/('controller_'+args.group+'_receipt.json');fd=os.open(receipt,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
 state={'status':'root_explicit_execution_started','group':args.group,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'collector_invocations':0,'maximum_collector_invocations':1,'qualified_skips':[j['job_id']for j in skips],'steps':[],'binary_scratch_retention':'canonical C21 full SHA/size manifest then disposable NEWscratch cleanup','scope':'Raw outcomes only; root audits/adopts. No actualphase/fullROI/causal certificate.'};save(receipt,state)
 # Explicit default cleanup enables serial bounded storage. No history deleted.
 try:
  need(shutil.disk_usage('/tmp').free>=review['storage']['minimum_free_bytes'][args.group],'Scientific payload planning allowance plus 2GiB reserve insufficient')
  batches=[pending]if pending else []
  for batch in batches:
   generations(review);verify(review['tool_bindings']);verify(review['packet_bindings'])
   still=[]
   for j in batch:
    if current(j,root)=='qualified_skip':continue
    verify(j['readonly_bindings']);need(shutil.disk_usage('/tmp').free>=2*1024**3+512*1024**2+load(j['prepared_plan']['path']).get('runtime_asset_bytes',0),'Fresh attempt reserve insufficient');stage(j);still.append(j)
   if not still:continue
   argv=still[0]['argv'];need('--retain-auxiliary'not in argv and'--skip-qualified-selected'in argv,'Canonical disposable scratch/skip gate absent')
   # Legacy C21 discovers ncu through PATH; pin exact directory and binary SHA.
   env=os.environ.copy();env['PATH']='/usr/lib/nsight-compute:/usr/bin:/bin';env['LD_PRELOAD']='';env['LD_LIBRARY_PATH']='';log=HERE/(args.group+'_'+str(len(state['steps']))+'_collector.log');need(not log.exists(),'Refuse historical child log')
   row={'jobs':[j['job_id']for j in still],'argv':argv,'status':'invoking_exact_collector'};state['steps'].append(row);state['collector_invocations']+=1;need(state['collector_invocations']<=state['maximum_collector_invocations'],'Finite bound exceeded');save(receipt,state)
   with log.open('xb')as out:done=subprocess.run(argv,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
   row.update(returncode=done.returncode,log={'path':str(log),'sha256':sha(log)},outcomes=[dict(job_id=j['job_id'],**outcome(j))for j in still]);save(receipt,state);verify(review['tool_bindings'])
   need(done.returncode==0,'Collector command failed; no automatic retry')
   if any((o.get('profile')or{}).get('stop_reason')=='external_gpu_activity_detected_own_profile_stopped'for o in row['outcomes']):state['status']='new_foreign_activity_stop_remaining_jobs_unattempted_no_retry';break
  else:state['status']='finite_traversal_returned_root_admission_required'
 except BaseException as e:state.update(status='held_or_failed_no_automatic_retry',error=repr(e));raise
 finally:state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(receipt,state)
if __name__=='__main__':main()
