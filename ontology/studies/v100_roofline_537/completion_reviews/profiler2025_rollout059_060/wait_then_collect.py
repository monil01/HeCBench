#!/usr/bin/env python3
"""Private root-reviewed finite 059/060 proposal. No launch without --execute.
No GPU querying, signalling, compilation or profiling by this controller itself:
only the exact adopted collector owns GPU6 lock/guard/idle/profile operations.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
 q=p.with_suffix('.new');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def verify(bindings):
 for p,h in bindings.items():need(Path(p).is_file()and digest(p)==h,'Readonly binding absent/changed '+p)
def generation(e):
 p=Path('/proc')/str(e['pid'])
 try:
  state=(p/'stat').read_text().rsplit(')',1)[1].split()
  if state[19]!=e['start_ticks']:return 'original_generation_finished_PID_reused_no_signal'
  need(p.stat().st_uid==e['uid']==os.getuid(),'Predecessor ownership mismatch')
  if state[0]=='Z':return 'original_generation_finished_zombie'
  need(e['argv']is not None,'Unexpected live generation previously absent; root review required')
  argv=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x]
  need(argv==e['argv'],'Predecessor argv changed')
  return 'same_owned_generation_live'
 except(FileNotFoundError,ProcessLookupError):return 'original_generation_finished_absent'
def deployed(review):
 verify(review['future_collector_tool_bindings'])
 method=review['method_review'];p=Path(method['path']);need(p.is_file()and digest(p)==method['sha256'],'Exact root-admitted method not deployed')
 doc=json.loads(p.read_bytes());need(doc['status']=='root_approved_after_saved_calibration_pass'and doc['bundle_sha256']==method['bundle_sha256'],'Admitted method identity mismatch')
 bundle=p.parent/doc['bundle_file'];need(bundle.is_file()and digest(bundle)==method['bundle_sha256'],'Exact method bundle not deployed')
 spec=review['profiler'];need(doc['profiler']==spec,'Profiler method review mismatch')
 verify({spec['executable']:spec['executable_sha256'],spec['target_ELF']:spec['target_ELF_sha256']})

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--review-sha256',required=True);ap.add_argument('--maximum-predecessor-wait-seconds',type=int,default=86400);a=ap.parse_args()
 need(1<=a.maximum_predecessor_wait_seconds<=86400,'Finite predecessor waiting must be<=86400')
 p=HERE/'launch_review.json';need(digest(p)==a.review_sha256,'Exact root-reviewed launcher SHA required');review=json.loads(p.read_bytes());need(digest(__file__)==review['controller_sha256'],'Controller source drift')
 need(len(review['jobs'])==2 and[case['benchmark']for case in review['jobs']]==['miniDGS','atomicSystemWide'],'Exact two jobs required')
 verify(review['readonly_bindings'])
 if not a.execute:print(json.dumps({'not_executed':True,'jobs':review['jobs'],'gates':['root newmethoddeployment','exact predecessor generations completed','collector skipqualified + all strict source/method/GPU gates']}));return
 receipt=HERE/'controller_receipt.json';need(not receipt.exists(),'One-shot receipt already exists; no duplicated launch/resume by reconstruction')
 state={'status':'waiting_predecessors','started_at':utc(),'review_sha256':a.review_sha256,'controller_sha256':review['controller_sha256'],'collector_invocations':0,'maximum_collector_invocations':2,'steps':[],'signals':0,'GPU_queries':0,'qualification':'Completion records only; no scientific/fullROI/causal certificate. Original5checks mandatory in collector. Idle/foreign waiting distinct from per-model timeout.'};save(receipt,state);until=time.monotonic()+a.maximum_predecessor_wait_seconds
 try:
  while True:
   frontier=[dict(e,observed_status=generation(e))for e in review['predecessors']];state['predecessor_observations']=frontier;save(receipt,state)
   if all(x['observed_status']!='same_owned_generation_live'for x in frontier):break
   need(time.monotonic()<until,'Finite predecessor waiting expired; no launches');time.sleep(min(5,max(0,until-time.monotonic())))
  # Deployment is checked only after old-generation work has ended. Never
  # install/copy tools or substitute a proposed collector automatically.
  deployed(review);verify(review['readonly_bindings'])
  for case in review['jobs']:
   need(all(generation(e)!='same_owned_generation_live'for e in review['predecessors']),'Predecessor generation unexpectedly live')
   deployed(review);verify(review['readonly_bindings']);argv=case['argv']
   need('--skip-qualified-selected'in argv and'--profiler-method-review-sha256'in argv,'No skip-qualified/method gate')
   log=HERE/(case['benchmark']+'_collector.log');need(not log.exists(),'Refuse existing child log')
   row={'benchmark':case['benchmark'],'argv':argv,'started_at':utc(),'status':'invocation_starting'};state['steps'].append(row);state['collector_invocations']+=1;save(receipt,state)
   with log.open('xb')as out:result=subprocess.run(argv,cwd=review['source_root'],stdout=out,stderr=subprocess.STDOUT)
   row.update(status='collector_returned_inspect_individual_actual_outcomes',returncode=result.returncode,finished_at=utc(),log={'path':str(log),'sha256':digest(log)});save(receipt,state)
   deployed(review);verify(review['readonly_bindings'])
  state['status']='finite_two_job_traversal_returned_not_admission_certificate'
 except BaseException as error:state.update(status='held_or_failed_no_automatic_retry',error=repr(error));raise
 finally:state['finished_at']=utc();save(receipt,state)
if __name__=='__main__':main()
