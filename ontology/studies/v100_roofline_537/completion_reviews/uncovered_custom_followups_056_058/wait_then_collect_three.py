#!/usr/bin/env python3
"""PRIVATE finite proposal. Root review/explicit --execute required; never run by staging.
Three sequential source-equivalent custom collectors, each at most once. No
signals/GPU queries here; frozen collector owns GPU6-only lock/guard/idle checks.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,time
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 q=p.with_suffix('.new');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def verify(bindings):
 for p,h in bindings.items():
  if not Path(p).is_file()or sha(p)!=h:raise RuntimeError('Readonly binding drift '+p)
def predecessor(e):
 p=Path('/proc')/str(e['pid'])
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split()
  if st[19]!=e['start_ticks']:return 'generation_finished_PID_reused_no_signal'
  if e.get('already_finished_or_reused'):raise RuntimeError('Unexpected resurrected original generation')
  if p.stat().st_uid!=e['uid']or e['uid']!=os.getuid():raise RuntimeError('Predecessor ownership mismatch')
  if st[0]=='Z':return 'generation_finished_zombie'
  args=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x]
  if args!=e['argv']:raise RuntimeError('Predecessor argv changed')
  return 'same_owned_generation_live'
 except(FileNotFoundError,ProcessLookupError):return 'generation_finished_absent'
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--review-sha256',required=True);ap.add_argument('--maximum-predecessor-wait-seconds',type=int,default=86400);a=ap.parse_args()
 if not 1<=a.maximum_predecessor_wait_seconds<=86400:raise RuntimeError('Finite predecessor wait required')
 path=HERE/'launch_review.json'
 if sha(path)!=a.review_sha256:raise RuntimeError('Exact root-reviewed launch SHA required')
 review=json.loads(path.read_text())
 if sha(__file__)!=review['controller_sha256']:raise RuntimeError('Controller source drift')
 verify(review['readonly_bindings'])
 if len(review['cases'])!=3 or [x['benchmark']for x in review['cases']]!=['assert','atomicSystemWide','columnarSolver']:raise RuntimeError('Exact three-case order required')
 if not a.execute:print(json.dumps({'not_executed':True,'cases':review['cases']}));return
 receipt=HERE/'future_controller_receipt.json'
 if receipt.exists():raise RuntimeError('Immutable one-shot receipt exists; no automatic retry')
 state={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_sha256':a.review_sha256,'maximum_collector_invocations':3,'actual_collector_invocations':0,'steps':[],'signals':0,'GPU_queries_by_controller':0,'qualification':'Predecessor wait finite; frozen collector idle waits remain outside execution deadline and require finite root supervision. Returncode alone is never certification.'}
 save(receipt,state);until=time.monotonic()+a.maximum_predecessor_wait_seconds
 try:
  for case in review['cases']:
   item={'benchmark':case['benchmark'],'wait_for':case['wait_original_controller'],'status':'waiting_original_generation'};state['steps'].append(item);save(receipt,state)
   while True:
    status=predecessor(case['wait_original_controller']);item['predecessor_status']=status
    if status!='same_owned_generation_live':break
    if time.monotonic()>=until:raise RuntimeError('Finite total predecessor wait expired; no further launch')
    time.sleep(min(5,max(0,until-time.monotonic())))
   verify(review['readonly_bindings'])
   ncu=shutil.which('ncu');expected=review['NCU_wrapper']
   if ncu is None or str(Path(ncu).resolve())!=expected['path']or sha(ncu)!=expected['sha256']:raise RuntimeError('Old NCU PATH resolution drift; no PATH rewrite/fallback')
   log=HERE/('future_'+case['benchmark']+'_collector.log')
   item.update(status='collector_invocation_starting',argv=case['argv'],started_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(receipt,state)
   with log.open('xb')as out:
    state['actual_collector_invocations']+=1;save(receipt,state)
    result=subprocess.run(case['argv'],cwd=review['source_root'],stdout=out,stderr=subprocess.STDOUT)
   item.update(status='collector_returned_inspect_actual_permodel_evidence',returncode=result.returncode,log={'path':str(log),'sha256':sha(log)},finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(receipt,state)
   verify(review['readonly_bindings'])
  state['status']='finite_three_case_traversal_returned_not_scientific_certificate'
 except BaseException as error:state.update(status='held_or_failed_no_automatic_retry',error=repr(error));raise
 finally:state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(receipt,state)
if __name__=='__main__':main()
