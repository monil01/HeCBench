#!/usr/bin/env python3
"""Finite, resumable root-adopted V100 completion orchestration.

Creating/reviewing this file executes nothing. Only an explicit --execute starts
commands; no full campaign/master is launched. Existing root collectors are
identified by PID plus Linux start ticks. Never signals any process. Queue end
means ready for root review, not complete ROI/science/causal coverage.
"""
import argparse
import collections
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MODELS=('cuda','cuda_jl','ka','jacc')
STOP_REASONS={'external_gpu_activity_detected_own_profile_stopped','wall_timeout'}
CONTROLLERS={'v100_roofline_campaign.py','v100_roofline_followups.py',
 'v100_roofline_collect.py','v100_roofline_retry.py','v100_roofline_handoff.py'}
TOOLS=('v100_roofline_summarize.py','v100_roofline_audit.py','v100_roofline_retry.py',
 'v100_roofline_refresh.py','export_v100_measurement_failures.py',
 'export_v100_benchmark_assessment.py','export_v100_retained_roi.py')

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.next')
 with tmp.open('w') as f:
  json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def proc(pid):
 try:
  p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'start_ticks':fields[19],'state':fields[0],'uid':p.stat().st_uid,
   'command':[(x.decode(errors='replace')) for x in (p/'cmdline').read_bytes().split(b'\0')if x]}
 except (FileNotFoundError,ProcessLookupError):return None
def live(identity):
 p=proc(identity['pid'])
 return p if p and p['start_ticks']==identity['start_ticks'] and p['state']!='Z' else None
def owned_directory(p,under_tmp=False):
 if p.is_symlink() or not p.is_dir() or p.stat().st_uid!=os.getuid():raise RuntimeError('Existing owned directory required: '+str(p))
 if under_tmp and (not p.resolve().is_relative_to('/tmp') or p.resolve()==Path('/tmp')):raise RuntimeError('Owned directory strictly within /tmp required')
def controller(value,custom_reviews):
 pieces=value.split(':')
 if len(pieces)!=2 or not all(x.isdigit() and int(x)>0 for x in pieces):raise ValueError('Controller must be PID:START_TICKS')
 ident={'pid':int(pieces[0]),'start_ticks':pieces[1]};p=live(ident)
 if ident['pid']==os.getpid():raise ValueError('Cannot monitor self')
 if p:
  names={Path(x).name for x in p['command']}
  if p['uid']!=os.getuid() or 'v100_roofline_refresh.py' in names or 'v100_roofline_completion.py' in names:raise ValueError('Only explicitly owned GPU controllers; never refresh/completion')
  if not(names&CONTROLLERS):
   proof=custom_reviews.get(value,{})
   raw=(Path('/proc')/str(p['pid'])/'cmdline').read_bytes()
   if proof.get('source_reviewed_GPU_controller')is not True or not proof.get('reason') or proof.get('command_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('Custom controller requires exact command SHA review; never auto-discover custom jobs')
   ident['custom_controller_review']=proof
  ident['initial_observation']=p
 else:ident['initial_observation']='already absent, exited, zombie or PID reused; no signal/no historical identity claim'
 return ident

def worker(specpath):
 """Independent durable command receipt survives supervisor disconnect/death."""
 specpath=Path(specpath);s=load(specpath);receipt=Path(s['receipt'])
 if receipt.exists():raise RuntimeError('Worker receipt already exists; never repeat step')
 if sha(__file__)!=s['supervisor_sha256']:raise RuntimeError('Supervisor source changed')
 for p,h in s['tool_bindings'].items():
  if sha(p)!=h:raise RuntimeError('Bound tool/review changed '+p)
 state={'spec_sha256':sha(specpath),'started_at':utc(),'worker':proc(os.getpid()),'command':s['command'],
  'complete':False,'returncode':None,'child':None,'role':s['role'],'log':s['log']}
 save(receipt,state)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='6' if s['role']=='finite_operational_retry' else ''
 try:
  with Path(s['log']).open('xb') as f:
   child=subprocess.Popen(s['command'],cwd=s['source_root'],env=env,stdout=f,stderr=subprocess.STDOUT)
   state['child']=proc(child.pid);save(receipt,state)
   # No operational timeout here: foreign-GPU wait may legitimately be unbounded.
   state['returncode']=child.wait()
 except Exception as error:state['exception']=repr(error)
 state.update(complete=True,finished_at=utc())
 if Path(s['log']).exists():state['log_sha256']=sha(s['log'])
 save(receipt,state)
 return 0 if state['returncode']==0 else 1

class Supervisor:
 def __init__(self,a):
  self.a=a;self.root=a.source_root.resolve();self.raw=a.raw_root.resolve();self.out=a.output_root.resolve();self.study=self.root/'ontology/studies/v100_roofline_537';self.children=[]
  owned_directory(a.source_root);owned_directory(a.raw_root,True)
  if a.output_root.is_symlink() or not self.out.is_relative_to(self.raw) or self.out==self.raw:raise RuntimeError('Private completion output must be a fresh child of raw root')
  if not (self.root/'tools').is_dir() or not (self.study/'benchmarks').is_dir():raise RuntimeError('Required source/study missing')
  if not a.reviews.is_file():raise RuntimeError('Explicit approved reviews file required')
  custom=load(a.controller_reviews).get('controllers',{})if a.controller_reviews else {}
  self.controllers=[controller(x,custom)for x in a.controller]
  if not self.controllers:raise RuntimeError('Supply explicit controller PID:starttime identities (including all live GPU followups/custom controllers)')
  if len({(x['pid'],x['start_ticks'])for x in self.controllers})!=len(self.controllers):raise RuntimeError('Duplicate controller identity')
  if not 0<=a.max_retry_rounds<=2 or a.poll_seconds<5 or a.timeout<1 or a.freeze_every<1:raise RuntimeError('Invalid finite bounds')
  self.bindings={str(self.root/'tools'/n):sha(self.root/'tools'/n)for n in TOOLS}
  if a.controller_reviews:self.bindings[str(a.controller_reviews.resolve())]=sha(a.controller_reviews)
  self.bindings[str(a.reviews.resolve())]=sha(a.reviews);self.bindings[str(Path(__file__).resolve())]=sha(__file__)
  self.config={'schema_version':1,'source_root':str(self.root),'raw_root':str(self.raw),'output_root':str(self.out),
   'controllers':[{'pid':x['pid'],'start_ticks':x['start_ticks']}for x in self.controllers],
   'retry_batch_ids':['039','041'][:a.max_retry_rounds],'reviews':str(a.reviews.resolve()),'max_retry_rounds':a.max_retry_rounds,
   'timeout':a.timeout,'freeze_every':a.freeze_every,'tool_bindings':self.bindings,'existing_round_states':a.existing_round_state,
   'claims':{'complete_GPU_ROI':False,'causal_classification_complete':False,'scientific_complete':False,'automatic_git_or_push':False}}
  if self.out.exists():
   if not a.resume:raise RuntimeError('Fresh output required; explicit --resume required for existing journal')
   owned_directory(self.out,True)
   if load(self.out/'configuration.json')!=self.config:raise RuntimeError('Resume inputs/tool hashes changed; preserve old journal and review explicitly')
  else:
   if a.resume:raise RuntimeError('Cannot resume nonexistent checkpoint')
   self.out.mkdir();save(self.out/'configuration.json',self.config)
  self.campaign_lock=(self.raw/'finite_completion_supervisor.lock').open('a');fcntl.flock(self.campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.lock=(self.out/'supervisor.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.journal=self.out/'journal.jsonl';self.checkpoint=self.out/'checkpoint.json'
  self.state=load(self.checkpoint)if self.checkpoint.exists()else{'schema_version':1,'status':'prepared','steps':{},'rounds':{},'created_at':utc()}
  self.existing={}
  for x in a.existing_round_state:
   k,sep,v=x.partition('=')
   if not sep or k not in ['039','041'] or v not in ['finished','preserve']:raise RuntimeError('--existing-round-state ID=finished|preserve only; never040')
   if k in self.existing:raise RuntimeError('Duplicate existing round declaration')
   self.existing[k]=v
 def note(self,event,**value):
  r={'time':utc(),'event':event,**value}
  with self.journal.open('a')as f:f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  self.state['last_event']=r;save(self.checkpoint,self.state);print(json.dumps(r),flush=True)
 def verify(self):
  for p,h in self.bindings.items():
   if sha(p)!=h:raise RuntimeError('Frozen supervisor tool/review changed: '+p)
 def wait_controllers(self):
  self.state['status']='waiting_explicit_owned_controllers';last=None
  while True:
   active=[]
   for ident in self.controllers:
    p=live(ident)
    if p:
     if p['uid']!=os.getuid():raise RuntimeError('Controller UID changed')
     active.append(p)
   if active!=last:self.note('controller_frontier',active=active,signal_sent=False);last=active
   if not active:return
   time.sleep(self.a.poll_seconds)
 def step(self,name,command,role='CPU_metadata_audit',allow_failure=False):
  self.verify();d=self.out/'steps'/name;info=self.state['steps'].get(name)
  if info is None:
   if d.exists():raise RuntimeError('Unjournaled existing step directory; manual review, never relaunch '+name)
   d.mkdir(parents=True);spec={'command':command,'role':role,'receipt':str(d/'receipt.json'),'log':str(d/'execution.log'),
    'source_root':str(self.root),'tool_bindings':self.bindings,'supervisor_sha256':sha(__file__)};save(d/'spec.json',spec)
   info={'status':'launch_intent','spec_sha256':sha(d/'spec.json'),'directory':str(d)};self.state['steps'][name]=info
   self.note('step_launch_intent',name=name,command=command,role=role)
   with(d/'worker_stdout.log').open('xb')as worker_log:
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--internal-worker',str(d/'spec.json')],cwd=self.root,start_new_session=True,stdout=worker_log,stderr=subprocess.STDOUT)
   self.children.append(child);info.update(status='launched',worker=proc(child.pid));self.note('step_launched',name=name,worker=info['worker'])
  if sha(d/'spec.json')!=info['spec_sha256'] or load(d/'spec.json')['command']!=command:raise RuntimeError('Step specification drift '+name)
  waited=0
  while not (d/'receipt.json').exists():
   if info.get('worker')and live(info['worker']):time.sleep(1);continue
   # Tiny startup grace for new worker; resume ambiguity never repeats a command.
   if waited<10:time.sleep(1);waited+=1;continue
   raise RuntimeError('Ambiguous interrupted launch; manual review, do not repeat '+name)
  while True:
   receipt=load(d/'receipt.json')
   if receipt['spec_sha256']!=info['spec_sha256']:raise RuntimeError('Worker receipt spec mismatch '+name)
   if receipt['complete']:break
   if not live(receipt['worker']):raise RuntimeError('Worker disappeared with incomplete command; no duplicate launch '+name)
   time.sleep(self.a.poll_seconds)
  if receipt.get('log_sha256') and sha(receipt['log'])!=receipt['log_sha256']:raise RuntimeError('Step log drift '+name)
  info.update(status='finished',returncode=receipt['returncode'],receipt_sha256=sha(d/'receipt.json'))
  self.note('step_finished',name=name,returncode=receipt['returncode'],receipt=str(d/'receipt.json'))
  if receipt['returncode']!=0 and not allow_failure:raise RuntimeError('Required step failed '+name)
  return receipt
 def tool(self,name,*args):return[sys.executable,str(self.root/'tools'/name),*map(str,args)]
 def freeze_audit(self,label):
  self.step(label+'_summarize',self.tool('v100_roofline_summarize.py','--source-root',self.root,'--raw-root',self.raw))
  target=self.out/(label+'_integrity_audit.json')
  self.step(label+'_integrity_audit',self.tool('v100_roofline_audit.py','--source-root',self.root,'--output',target))
  result=load(target)
  if not isinstance(result.get('issues'),list) or result['issues']:raise RuntimeError('Integrity audit must report exactly zero issues')
  self.note('zero_issue_integrity_gate',label=label,output=str(target),sha256=sha(target),issues=0)
 def frontier(self,label):
  lock=Path('/tmp/hecbench-roofline-metadata.lock')
  if not lock.is_file():raise RuntimeError('Existing coordinator metadata lock required')
  rows=[];inputs={}
  with lock.open('rb')as f:
   fcntl.flock(f,fcntl.LOCK_SH)
   for lp in sorted((self.study/'benchmarks').glob('*.json')):
    data=lp.read_bytes();ledger=json.loads(data);inputs[str(lp)]=hashlib.sha256(data).hexdigest()
    for m,s in ledger['models'].items():
     if s.get('profiling_status')!='profile_process_failed':continue
     ids=s.get('profile_ids',[])
     if not ids:continue
     p=self.root/ids[0]/'profile_provenance.json.gz'
     if not p.is_file():continue
     packed=p.read_bytes();pr=json.loads(gzip.decompress(packed));reason=pr.get('profile',{}).get('stop_reason')
     if reason in STOP_REASONS:rows.append({'benchmark':ledger['benchmark'],'model':m,'stop_reason':reason,'selected_profile_id':ids[0],'provenance_compressed_sha256':hashlib.sha256(packed).hexdigest()})
  report={'created_at':utc(),'slots':rows,'count':len(rows),'input_hashes':inputs,'eligible_reasons_only':sorted(STOP_REASONS),'all_other_failures_not_retried':True}
  path=self.out/(label+'_frontier.json')
  if path.exists():
   # Preserve the old frontier snapshot; resumed runs do not overwrite it.
   self.note('frontier_reobserved',label=label,current_count=len(rows),archived_sha256=sha(path))
  else:save(path,report)
  return report
 def round(self,batch):
  key='round_'+batch
  if self.state['rounds'].get(batch,{}).get('consumed'):self.note('round_already_consumed',batch_id=batch);return
  self.freeze_audit('before_'+batch);front=self.frontier('before_'+batch)
  journal=self.raw/('retry_'+batch+'.jsonl');frozen=self.raw/('retry_'+batch+'_frontier.json');roots=sorted(self.raw.glob('batch'+batch+'-*'))
  step_known=key in self.state['steps']
  if (journal.exists()or frozen.exists()or roots)and not step_known:
   declared=self.existing.get(batch)
   if declared is None:raise RuntimeError('Existing retry artifacts require explicit --existing-round-state '+batch+'=finished|preserve; no relaunch')
   events=[json.loads(x)for x in journal.read_text().splitlines()]if journal.exists()else[]
   finished=any(x.get('event')=='retry_traversal_finished'for x in events)
   if declared=='finished'and not finished:raise RuntimeError('No actual traversal-finished event for imported round '+batch)
   evidence={str(p):sha(p)for p in [journal,frozen]if p.is_file()}
   self.state['rounds'][batch]={'consumed':True,'existing_state':declared,'finished_event_present':finished,'preserved_artifact_hashes':evidence,'existing_plan_roots':[str(x)for x in roots]}
   self.note('existing_round_preserved_no_relaunch',batch_id=batch,state=self.state['rounds'][batch]);return
  if not front['count']and not step_known:
   self.state['rounds'][batch]={'consumed':True,'empty_eligible_frontier':True};self.note('finite_retry_round_empty',batch_id=batch);return
  command=self.tool('v100_roofline_retry.py','--source-root',self.root,'--raw-root',self.raw,'--batch-id',batch,'--reviews',self.a.reviews.resolve(),'--execute','--freeze-every',self.a.freeze_every,'--timeout',self.a.timeout)
  outcome=self.step(key,command,'finite_operational_retry',allow_failure=True)
  self.state['rounds'][batch]={'consumed':True,'recorded_launcher_returncode':outcome['returncode'],'no_duplicate_iteration':True}
  self.note('finite_retry_round_consumed',batch_id=batch,returncode=outcome['returncode'])
  # Even a partially failed traversal is frozen/audited, never silently restarted.
  self.freeze_audit('after_'+batch)
 def finish(self):
  self.state['status']='final_CPU_exports';self.note('final_refresh_started')
  command=self.tool('v100_roofline_refresh.py','--source-root',self.root,'--raw-root',self.raw,'--once')
  if self.a.current537_display_review_sha256:command+=['--current537-display-review-sha256',self.a.current537_display_review_sha256]
  rec=self.step('final_refresh_once',command)
  # refresh currently logs per-tool errors but can itself return0; enforce each.
  events=[]
  for line in Path(rec['log']).read_text(errors='replace').splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
  bad=[x for x in events if isinstance(x,dict)and'returncode'in x and x['returncode']!=0]
  if bad:save(self.out/'refresh_failed_steps.json',bad);raise RuntimeError('Final refresh contained failed substeps')
  if not any(x.get('event')=='CPU_refresh_supervisor_finished'for x in events if isinstance(x,dict)):raise RuntimeError('Final refresh lacks finished event')
  self.freeze_audit('final')
  failure=self.out/'measurement_failure_census';self.step('failure_census_export',self.tool('export_v100_measurement_failures.py','--source-root',self.root,'--output-root',failure))
  fs=load(failure/'summary.json')
  if fs['integrity']['issues']or fs.get('live_ledger_drift_paths'):raise RuntimeError('Failure census integrity/drift gate failed')
  family=self.out/'retained_family_audit';self.step('retained_family_export',self.tool('export_v100_retained_roi.py','--source-root',self.root,'--output-root',family))
  assessment=self.out/'all537_assessment';self.step('all537_assessment_export',self.tool('export_v100_benchmark_assessment.py','--source-root',self.root,'--output-root',assessment,'--retained-family-audit',family/'audit.json'))
  data=load(assessment/'assessment.json')
  if data['registry_assessment_complete']is not True or data['denominators']['model_slots']!=2148 or len(data['model_assessments'])!=2148:raise RuntimeError('All537x4 assessment incomplete')
  issues=[{'benchmark':r['benchmark'],'model':r['model'],'issues':r['issues']}for r in data['model_assessments']if r.get('issues')]
  save(self.out/'assessment_issues.json',issues)
  if issues:raise RuntimeError('Assessment has unresolved evidence issues')
  remaining=[];permodel={}
  for m in MODELS:
   rr=[x for x in data['model_assessments']if x['model']==m]
   if len(rr)!=537:raise RuntimeError('Model denominator not537')
   counts=collections.Counter(x['profiling_status']for x in rr)
   permodel[m]={'denominator':537,'selected_status_counts':dict(counts),'assessment_categories':dict(collections.Counter(x['assessment_category']for x in rr))}
   for r in rr:
    status=r['profiling_status']
    if status not in ['profile_process_completed','roofline_not_applicable_no_device_kernels']:
     category=('hardware'if status=='native_hardware_blocked'else'replay_method'if'replay'in status else'plan'if r.get('plan_gates')else'failed'if'failed'in status or'quality_unverified'in status else'pending_or_other')
     remaining.append({'benchmark':r['benchmark'],'model':m,'status':status,'category':category,'plan_gates':r.get('plan_gates',[]),'selected_profile_ids':r.get('selected_profile_ids',[])})
  self.frontier('final_remaining_eligible')
  report={'schema_version':1,'status':'ready_for_root_review','created_at':utc(),'per_model':permodel,'remaining_counts':dict(collections.Counter(x['category']for x in remaining)),'remaining_records':remaining,'retry_rounds':self.state['rounds'],'integrity_issues':0,'audit':{'path':str(self.out/'final_integrity_audit.json'),'sha256':sha(self.out/'final_integrity_audit.json')},'failure_census_summary_sha256':sha(failure/'summary.json'),'assessment_sha256':sha(assessment/'assessment.json'),'limitations':['Queue traversal is not full GPU ROI coverage.','No scientific completion or causal bottleneck claim.','No gates, source mathematics or correctness checks waived.','Primary five-run timing series unchanged.','Foreign GPU wait was unbounded; no foreign processes signaled.','No automatic Git commit, push or publication.']}
  save(self.out/'ready_for_root_review.json',report);self.state['status']='ready_for_root_review';self.note('ready_for_root_review',remaining_counts=report['remaining_counts'])
 def run(self):
  self.note('completion_supervisor_started',pid=os.getpid(),execute=self.a.execute,controllers=self.controllers)
  if not self.a.execute:
   self.state['status']='proposal_only_no_commands_executed';self.note('proposal_only',planned_retry_ids=self.config['retry_batch_ids']);return
  self.wait_controllers()
  # Reconcile durable children BEFORE any new metadata work after resume.
  for name,info in list(self.state['steps'].items()):
   if info.get('status')!='finished':
    spec=load(Path(info['directory'])/'spec.json')
    self.step(name,spec['command'],spec['role'],allow_failure=spec['role']=='finite_operational_retry')
  for batch in self.config['retry_batch_ids']:self.round(batch)
  self.finish()

def main():
 if len(sys.argv)==3 and sys.argv[1]=='--internal-worker':return worker(sys.argv[2])
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--reviews',type=Path,required=True);ap.add_argument('--controller',action='append',required=True,help='Explicit owned GPU controller PID:Linux_START_TICKS; repeat all main/followup/custom controllers; never CPU refresh');ap.add_argument('--controller-reviews',type=Path,help='Optional JSON controllers keyed PID:START with command_sha256 of raw /proc/PID/cmdline, source_reviewed_GPU_controller:true and reason; never CPU refresh');ap.add_argument('--max-retry-rounds',type=int,default=2);ap.add_argument('--existing-round-state',action='append',default=[],help='039=finished|preserve or041=finished|preserve; requires archived proof for finished');ap.add_argument('--poll-seconds',type=int,default=15);ap.add_argument('--timeout',type=int,default=600);ap.add_argument('--freeze-every',type=int,default=10);ap.add_argument('--current537-display-review-sha256',help='Exact approved current537 display authority SHA; forwarded only to final CPU refresh.');ap.add_argument('--resume',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args();s=None
 try:s=Supervisor(a);s.run();return 0
 except Exception as error:
  if s is not None:
   s.state['status']='blocked_requires_root_review';s.note('supervisor_failed_closed',error=repr(error),no_automatic_repeat=True)
  else:print(json.dumps({'status':'not_started','error':repr(error)}),file=sys.stderr)
  return 1
if __name__=='__main__':raise SystemExit(main())
