#!/usr/bin/env python3
"""Private finite072 continuation. Root owns new exact authorization and --execute.
No GPU query/signals; only exact C21 collector owns GPU6 idle/lock/guard/processes.
"""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,shutil,subprocess,fcntl,time,csv
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
 for p,h in plan['execution_bindings'].items():need(sha(p)==h,'Staged execution binding differs')
 dst.write_bytes(Path(job['prepared_plan']['path']).read_bytes());need(sha(dst)==job['prepared_plan']['sha256'],'Staged plan differs')

def outcome(job):
 slot=Path(job['future_plan_path']).parent;p=slot/'profile_provenance.json'
 if not p.exists():return {'status':'no_provenance_inspect_log'}
 d=load(p);r={'status':d['status'],'profile':d.get('profile'),'provenance_sha256':sha(p),'auxiliary_manifest_sha256':d.get('auxiliary_manifest_sha256'),'binary_payloads_retained':False,'not_scientific_admission':True}
 if job['benchmark']=='tgvnn'and d['status']=='profile_process_completed':
  log=slot/'execution.log';rows=[json.loads(l[len('STUDY_TRIAL '):])for l in log.read_text().splitlines()if l.startswith('STUDY_TRIAL ')]
  need([x['trial']for x in rows]==[1,2,3,4,5],'TGVNN requires all5actual checks')
  need(all(x['correct']is True and x['checked_values']==72089820 and x.get('timing_valid')is True and x.get('input_contract_matched')is True and x.get('max_independent_field_budget_ratio',2)<=1 for x in rows),'TGVNN scientific/check flags incomplete')
  mp=slot/'profiling_auxiliary_manifest.json';need(sha(mp)==d['auxiliary_manifest_sha256'],'TGVNN full scratch manifest SHA mismatch');entries={e['path']:e for e in load(mp)}
  for i in range(1,6):
   e=entries['profile_outputs/trial'+str(i)+'.bin'];need(e['bytes']==288359280 and len(e['sha256'])==64,'TGVNN five complete primary payload manifest missing')
  need(not Path(job['scratch']).exists(),'Canonical cleanup unexpectedly did not finish')
  r['five_complete_primary_manifest_and_checks_verified']=True
 return r

UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
RAW=Path('/tmp/hecbench-roofline-v100-20261020')
def stable_idle(log):
 """Prospective root execution only: GPU6 scoped read-only queries under existing GPU lock.
 Release before invoking C21, which independently reacquires/rechecks ownership.
 """
 since=None;samples=0;last=None
 with Path('/tmp/hecbench-julia-study-gpu.lock').open('r')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with log.open('x')as out:
   while True:
    need(shutil.disk_usage('/tmp').free>=2*1024**3,'Storage reserve below2GiB during idle wait')
    results=[]
    for fields in ['--query-gpu=uuid,memory.used,utilization.gpu','--query-compute-apps=pid']:
     cmd=['/usr/bin/nvidia-smi','-i',UUID,fields,'--format=csv,noheader,nounits'];proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
     results.append({'command':cmd,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr});need(proc.returncode==0,'GPU6 idle-query failure; no launch')
    data=list(csv.reader(results[0]['stdout'].splitlines()));need(len(data)==1 and len(data[0])==3,'Unknown GPU6 identity/idle record');u,mem,util=[v.strip()for v in data[0]];need(u==UUID,'GPU6 UUID changed');mem=int(mem);util=int(util)
    idle=mem==0 and util==0 and not results[1]['stdout'].strip();now=time.monotonic()
    if idle:
     since=now if since is None else since;samples+=1
    else:since=None;samples=0
    event={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU6_only':True,'results':results,'idle':idle,'consecutive_samples':samples,'stable_seconds':now-since if since is not None else 0};out.write(json.dumps(event)+'\n');out.flush();os.fsync(out.fileno());last=event
    if since is not None and samples>=6 and now-since>=10:return {'path':str(log),'sha256':sha(log),'stable_seconds':now-since,'samples':samples,'last':last,'lock_released_before_collector':True}
    time.sleep(2)

def main():
 ap=argparse.ArgumentParser(description='Root-reviewed finite26 UNATTEMPTED continuation; no retry or atomicCUDAjl');ap.add_argument('--execute',action='store_true');ap.add_argument('--authorization',type=Path,required=True);ap.add_argument('--authorization-sha256',required=True);a=ap.parse_args()
 need(sha(a.authorization)==a.authorization_sha256,'New exact root authorization required');auth=load(a.authorization);need(auth.get('approved')is True and auth.get('delivery_sha256')==sha(HERE/'DELIVERY.json'),'Root did not approve exact new packet');need(auth.get('GPU5_authorized')is False and auth.get('allow_new_scratch_manifest_then_cleanup')is True,'Explicit GPU6-only/storage authority missing')
 delivery=load(HERE/'DELIVERY.json')
 for name,e in delivery['files'].items():need(sha(HERE/name)==e['sha256']and(HERE/name).stat().st_size==e['bytes'],'Packet changed: '+name)
 review=load(HERE/'proposal072.json');need(sha(__file__)==review['controller_sha256'],'Controller drift');verify(review['tool_bindings']);verify(review['packet_bindings']);root=Path(review['source_root']);jobs=review['jobs'];need(len(jobs)==26 and len({j['job_id']for j in jobs})==26,'Exact26 job set required');need(all(j['job_id']!='atomicSystemWide__cuda_jl'and j['benchmark']!='tgvnn'for j in jobs),'Attempted atomic or independentTGVNN forbidden');need(set(auth.get('allowed_job_ids',[]))=={j['job_id']for j in jobs},'Root exact26 authorization mismatch')
 # Parent ended receipt and exact root-recorded generation are immutable bindings.
 generations(review)
 pending=[];skips=[]
 for j in jobs:
  need(Path(j['plan_root'])==RAW/('batch072-'+j['job_id']),'Nonflat or unreserved raw root');need(not Path(j['plan_root']).exists(),'Already-created072 root: preserve/hold, never resume');need(not Path(j['former065_future_plan_path']).parent.exists(),'Old065 job was staged/attempted; no continuation allowed');verify(j['readonly_bindings']);need(sha(j['prepared_plan']['path'])==j['prepared_plan']['sha256'],'Plan changed');(skips if current(j,root)=='qualified_skip'else pending).append(j)
 if not a.execute:print(json.dumps({'not_executed':True,'pending':[j['job_id']for j in pending],'qualified_skips':[j['job_id']for j in skips],'maximum_collector_invocations':26,'no_GPU_queries':True}));return
 receipt=HERE/'controller072_receipt.json';fd=os.open(receipt,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd);ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
 state={'status':'root_authorized_new26_traversal','controller_pid':os.getpid(),'controller_start_ticks':ticks,'authorization':str(a.authorization),'authorization_sha256':a.authorization_sha256,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'maximum_collector_invocations':26,'collector_invocations':0,'qualified_skips':[j['job_id']for j in skips],'steps':[],'parent065_restarted':False,'atomicCUDAjl_retried':False,'binary_payloads_retained':False};save(receipt,state)
 try:
  for j in pending:
   generations(review);verify(review['tool_bindings']);verify(review['packet_bindings'])
   if current(j,root)=='qualified_skip':state['qualified_skips'].append(j['job_id']);save(receipt,state);continue
   need(not Path(j['plan_root']).exists(),'Unexpected existing072 root');verify(j['readonly_bindings']);need(shutil.disk_usage('/tmp').free>=2*1024**3+512*1024**2+load(j['prepared_plan']['path']).get('runtime_asset_bytes',0),'Reserve insufficient')
   idle=stable_idle(HERE/('idle_'+j['job_id']+'.jsonl'));verify(j['readonly_bindings']);verify(review['tool_bindings'])
   if current(j,root)=='qualified_skip':state['qualified_skips'].append(j['job_id']);save(receipt,state);continue
   stage(j);argv=j['argv'];need('--retain-auxiliary'not in argv and'--skip-qualified-selected'in argv,'Reviewed C21 cleanup/skip flags changed')
   row={'job_id':j['job_id'],'status':'calling_exact_collector_once','argv':argv,'stable_idle':idle};state['steps'].append(row);state['collector_invocations']+=1;need(state['collector_invocations']<=26,'Finite bound exceeded');save(receipt,state)
   env=os.environ.copy();env.update(PATH='/usr/lib/nsight-compute:/usr/bin:/bin',LD_PRELOAD='',LD_LIBRARY_PATH='');log=HERE/('collector_'+j['job_id']+'.log')
   with log.open('xb')as out:proc=subprocess.run(argv,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
   observed=outcome(j);row.update(returncode=proc.returncode,log={'path':str(log),'sha256':sha(log)},outcome=observed);save(receipt,state);verify(review['tool_bindings']);need(proc.returncode==0,'Collector failed, no retry')
   if(observed.get('profile')or{}).get('stop_reason')=='external_gpu_activity_detected_own_profile_stopped':state['status']='new_foreign_stop_remaining_unattempted_no_retry';break
  else:state['status']='finite26_traversal_returned_root_review_required'
 except BaseException as error:state.update(status='held_failed_no_automatic_retry',error=repr(error));raise
 finally:state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(receipt,state)
if __name__=='__main__':main()
