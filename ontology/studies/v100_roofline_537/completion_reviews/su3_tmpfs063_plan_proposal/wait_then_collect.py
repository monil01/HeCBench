#!/usr/bin/env python3
"""UNEXECUTED private batch063 proposal. Root approval/new frontier required.
Creates fresh tmpfs only after --execute plus two exact root-approved receipts.
Never installs tools, queries/signals GPUs, deletes raw data or launches a master.
One collector call, Julia3 only; native remains explicitly held.
"""
import argparse,datetime,fcntl,hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
MODELS=('cuda_jl','ka','jacc')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,obj):
 temp=path.with_suffix('.new');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def verify(bindings):
 for path,digest in bindings.items():need(Path(path).is_file()and sha(path)==digest,'Readonly binding changed '+path)
def generation(e):
 p=Path('/proc')/str(e['pid'])
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  if fields[19]!=e['start_ticks']:return 'original_generation_finished_PID_reused_no_signal'
  need(p.stat().st_uid==os.getuid()==e['uid'],'Predecessor owner mismatch')
  if fields[0]=='Z':return 'original_generation_finished_zombie'
  need(e['argv']is not None,'Historically absent generation unexpectedly live')
  actual=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x]
  need(actual==e['argv'],'Exact predecessor command changed')
  return 'same_owned_generation_live'
 except(FileNotFoundError,ProcessLookupError):return 'original_generation_finished_absent'
def selected(source):
 with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
  fcntl.flock(lock,fcntl.LOCK_SH)
  ledger=source/'ontology/studies/v100_roofline_537/benchmarks/su3.json'
  obj=json.loads(ledger.read_bytes());digest=sha(ledger)
  rows={m:{'status':obj['models'][m]['profiling_status'],
       'sampled_kernel_count':obj['models'][m].get('sampled_kernel_count',0)}for m in ('cuda',*MODELS)}
 return {'time':utc(),'ledger_sha256':digest,'models':rows}
def qualified(row):return row['status']=='profile_process_completed'and row['sampled_kernel_count']>0
def no_symlinks(path):
 p=Path(path)
 need(p.is_absolute()and'..'not in p.parts,'Absolute lexical path required')
 for q in [*reversed(p.parents),p]:need(not q.is_symlink(),'No symlink staging '+str(q))
 need(p.resolve()==p,'Path alias forbidden')
def mkdir_private(path):
 no_symlinks(path);path.mkdir(mode=0o700)
 st=path.stat();need(st.st_uid==os.getuid()and stat.S_IMODE(st.st_mode)==0o700,'Owned700 staging required')
def tool_gate(review):
 verify(review['future_tool_bindings'])
 found=shutil.which('ncu')
 need(found in review['legacy_method_bindings'] and str(Path(found).resolve())==review['legacy_profiler_wrapper'],'Inherited PATH must resolve a bound alias to exact legacy2022 ncu; no silent SDK/PATH substitution')
 verify(review['legacy_method_bindings'])
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--execute',action='store_true')
 ap.add_argument('--launch-review',type=Path,required=True);ap.add_argument('--launch-review-sha256',required=True)
 ap.add_argument('--tmpfs-review',type=Path,required=True);ap.add_argument('--tmpfs-review-sha256',required=True)
 ap.add_argument('--maximum-predecessor-wait-seconds',type=int,default=86400)
 a=ap.parse_args();need(1<=a.maximum_predecessor_wait_seconds<=86400,'Bounded predecessor wait required')
 need(sha(a.launch_review)==a.launch_review_sha256,'Launch review changed')
 need(sha(a.tmpfs_review)==a.tmpfs_review_sha256,'Tmpfs root review changed')
 review=json.loads(a.launch_review.read_bytes());storage=json.loads(a.tmpfs_review.read_bytes())
 need(sha(__file__)==review['controller_sha256'],'Follower source drift')
 verify(review['packet_bindings'])
 if not a.execute:
  print(json.dumps({'not_executed':True,'root_approval_required':True,'models':list(MODELS),'native_held':True,'maximum_collector_invocations':1}));return
 need(review.get('root_reviewed')is True and storage.get('root_reviewed')is True,'Both independent root approvals required')
 need(review['root_authorizes_fresh_tmpfs_staging']is True,'External fresh-staging authorization absent')
 need(storage['execution_permission']is True,'Explicit reviewed tmpfs execution permission absent')
 need(set(storage['models'])=={'su3/'+m for m in MODELS},'Only Julia3 approved; native held')
 need(review['storage_review_required_plan_sha256']=={k:v['fresh_plan_sha256']for k,v in storage['models'].items()},'Approved plan map changed')
 source=Path(review['source_root']);raw=Path(review['raw_root']);destination=Path(review['future_plan_root']);owned=Path(review['owned_tmpfs_root'])
 need(destination==raw/'batch063-su3-tmpfs','Reserved exact raw batch required')
 receipt=HERE/'controller_receipt.json';need(not receipt.exists(),'One-shot receipt exists; no reconstruction/relaunch')
 state={'started_at':utc(),'status':'waiting_exact_predecessors','collector_invocations':0,'maximum_collector_invocations':1,
        'native_held':True,'launch_review_sha256':a.launch_review_sha256,'tmpfs_review_sha256':a.tmpfs_review_sha256,
        'GPU_queries':0,'signals':0,'no_primary_timing_replacement':True,'no_primary_raw_deletion':True,
        'reviewed_plan_sha256':review['storage_review_required_plan_sha256'],
        'reviewed_tool_sha256':review['future_tool_bindings'],
        'packet_bindings':review['packet_bindings']}
 # Serialize this one exact plan-root owner; no GPU lock is taken here.
 with (raw/'batch063-su3-tmpfs_controller.lock').open('a')as ownerlock:
  fcntl.flock(ownerlock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(receipt,state)
  for name,data in [('effective_launch_review.json',a.launch_review.read_bytes()),('effective_tmpfs_review.json',a.tmpfs_review.read_bytes())]:
   with(HERE/name).open('xb')as out:out.write(data);out.flush();os.fsync(out.fileno())
  try:
   deadline=time.monotonic()+a.maximum_predecessor_wait_seconds
   while True:
    observations=[dict(x,observed=generation(x))for x in review['predecessors']]
    state['predecessor_observations']=observations;save(receipt,state)
    if all(x['observed']!='same_owned_generation_live'for x in observations):break
    need(time.monotonic()<deadline,'Predecessor wait expired without launch');time.sleep(5)
   verify(review['packet_bindings']);tool_gate(review)
   snapshot=selected(source);state['selected_before_staging']=snapshot;save(receipt,state)
   if all(qualified(snapshot['models'][m])for m in MODELS):
    state['status']='all_eligible_models_already_qualified_no_staging_no_launch';return
   need(not destination.exists()and not owned.exists(),'Fresh raw/tmpfs roots required; never reuse old paths')
   no_symlinks(destination);no_symlinks(owned)
   need(owned.parent==Path('/dev/shm')and str(owned)==review['owned_tmpfs_root'],'Exact reviewed external namespace required')
   need(os.statvfs('/tmp').f_bavail*os.statvfs('/tmp').f_frsize>=2*1024**3,'/tmp2GiB floor before staging')
   # No payload copies: eligible Julia targets retain canonical read-only sources/assets.
   # Root-approved collector policy performs full memory/cgroup/tmpfs admission before any child.
   mkdir_private(destination);mkdir_private(owned)
   created=[]
   for model in MODELS:
    parent=owned/model;mkdir_private(parent);scratch=parent/'scratch';mkdir_private(scratch)
    for sub in ('profile_outputs','ncu_tmp'):mkdir_private(scratch/sub)
    out=destination/'su3'/model;out.mkdir(parents=True,mode=0o700)
    original=HERE/'collector_plans'/'su3'/model/'plan.json';need(sha(original)==storage['models']['su3/'+model]['fresh_plan_sha256'],'Draft plan changed')
    with(out/'plan.json').open('xb')as f:f.write(original.read_bytes());f.flush();os.fsync(f.fileno())
    created.append({'path':str(out/'plan.json'),'sha256':sha(out/'plan.json'),'external_scratch':str(scratch),'source_copy_bytes':0})
   state.update(status='staged_no_children_started',created=created);save(receipt,state)
   verify(review['packet_bindings']);tool_gate(review)
   need(all(generation(x)!='same_owned_generation_live'for x in review['predecessors']),'Predecessor generation still live')
   argv=[review['python'],str(source/'tools/v100_roofline_collect_tmpfs.py'),'--source-root',str(source),
    '--plan-root',str(destination),'--benchmark','su3','--skip-qualified-selected','--timeout','3600',
    '--reserve-mib','2048','--retain-auxiliary','--large-artifacts',
    '--tmpfs-scratch-review',str(a.tmpfs_review.resolve()),'--tmpfs-scratch-review-sha256',a.tmpfs_review_sha256]
   state.update(status='single_collector_starting',collector_invocations=1,argv=argv);save(receipt,state)
   with(HERE/'collector_stdout.log').open('xb')as log:
    process=subprocess.Popen(argv,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    statfields=(Path('/proc')/str(process.pid)/'stat').read_text().rsplit(')',1)[1].split()
    state['collector_identity']={'pid':process.pid,'start_ticks':statfields[19],'argv':argv};save(receipt,state)
    code=process.wait()
   state.update(status='single_collector_returned_not_admission_certificate',returncode=code,
    stdout_sha256=sha(HERE/'collector_stdout.log'),selected_after=selected(source))
   tool_gate(review);verify(review['packet_bindings'])
   model_receipts={}
   for model in MODELS:
    profile=destination/'su3'/model/'profile_provenance.json'
    model_receipts[model]={'path':str(profile),'present':profile.is_file(),
      'sha256':sha(profile)if profile.is_file()else None,
      'status':json.loads(profile.read_bytes()).get('status')if profile.is_file()else 'no_durable_profile_provenance_not_a_zero_attempt_claim'}
   state['model_receipts']=model_receipts
   if code!=0 or any((row['present']and row['status']!='profile_process_completed')or(not row['present']and not qualified(state['selected_after']['models'][model]))for model,row in model_receipts.items()):
    state['status']='collector_rejected_or_receipt_missing_permanent_review_hold'
    save(HERE/'permanent_hold_receipt.json',state)
  except BaseException as error:
   state.update(status='held_or_failed_no_automatic_retry',error=repr(error))
   log=HERE/'collector_stdout.log'
   if log.is_file():state['full_stdout']={'path':str(log),'sha256':sha(log),'bytes':log.stat().st_size}
   save(HERE/'permanent_hold_receipt.json',state)
   raise
  finally:
   state['finished_at']=utc();save(receipt,state)
if __name__=='__main__':main()
