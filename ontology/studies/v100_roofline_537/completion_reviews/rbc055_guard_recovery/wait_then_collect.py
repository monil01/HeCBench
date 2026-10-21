#!/usr/bin/env python3
"""UNEXECUTED finite055 follower proposal. Root review/--execute required.
Wait only exact owned050generation; no signals/GPU queries in this follower.
The frozen collector subsequently owns GPU6 locks/idle/foreign/process/writechecks.
"""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def verify(bound):
 for p,h in bound.items():
  if not Path(p).is_file()or sha(p)!=h:raise RuntimeError('Readonlybindingdrift '+p)
def dump(p,x):
 temp=p.with_suffix('.new');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def predecessor(expected):
 proc=Path('/proc')/str(expected['pid'])
 try:
  st=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if st[19]!=expected['start_ticks']:return 'generation_finished_PIDreused_do_not_signal'
  if proc.stat().st_uid!=expected['uid']or expected['uid']!=os.getuid():raise RuntimeError('Predecessorownershipmismatch')
  args=[b.decode()for b in(proc/'cmdline').read_bytes().split(b'\0')if b]
  if st[0]=='Z':return 'generation_finished_zombie'
  if args!=expected['argv']:raise RuntimeError('Predecessorargvchanged_hold')
  return 'same_owned_generation_live'
 except(FileNotFoundError,ProcessLookupError):return 'generation_finished_absent'
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--review-sha256',required=True);ap.add_argument('--wait-seconds',type=int,default=86400);a=ap.parse_args()
 if not 1<=a.wait_seconds<=86400:raise RuntimeError('Finitewait1to86400required')
 path=HERE/'launch_review.json'
 if sha(path)!=a.review_sha256:raise RuntimeError('ExactindependentrootreviewSHAbindingrequired')
 review=json.loads(path.read_text())
 if sha(Path(__file__))!=review['follower_sha256']:raise RuntimeError('Followerchanged')
 verify(review['readonly_bindings'])
 if not a.execute:print(json.dumps({'not_executed':True,'wait_for':review['predecessor']['identity'],'command':review['collector_argv']}));return
 output=Path(review['receipt_path'])
 if output.exists():raise RuntimeError('Immutableone-shotfollowerreceiptalreadyexists')
 receipt={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_sha256':a.review_sha256,'maximum_wait_seconds':a.wait_seconds,'finite_custom_retry':True,'maximum_collector_invocations':1,'GPU_queries_by_follower':0,'foreign_signals':0,'root_review_reference':'ExactlaunchreviewSHArequired; rootownsapprovalandfrontieradoption'};dump(output,receipt)
 until=time.monotonic()+a.wait_seconds
 try:
  while True:
   status=predecessor(review['predecessor']);receipt['predecessor_status']=status
   if status!='same_owned_generation_live':break
   if time.monotonic()>=until:raise RuntimeError('Finiteownedpredecessorwaitexpired_no_launch')
   time.sleep(min(5,max(0,until-time.monotonic())))
  verify(review['readonly_bindings'])
  ncu=shutil.which('ncu')
  if ncu is None or str(Path(ncu).resolve())!=review['NCU_wrapper']['path']or sha(ncu)!=review['NCU_wrapper']['sha256']:raise RuntimeError('NCUresolutiondrift_hold_noPATHmutation')
  # Exactly one collector call; no retry/selection override/source rewrite. Its
  # own method checks still bind sources/dependencies under GPUlock before/after.
  receipt['collector_started_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt['collector_argv']=review['collector_argv'];dump(output,receipt)
  with Path(review['collector_log_path']).open('xb')as out:
   result=subprocess.run(review['collector_argv'],cwd=review['source_root'],stdout=out,stderr=subprocess.STDOUT)
  receipt['collector_returncode']=result.returncode;receipt['collector_log_sha256']=sha(review['collector_log_path']);receipt['meaning']='Finitequeue traversal only; rootmustinspect permodelstatus/actualfull5checks/markers/counters. No completionfromreturncodealone.'
 except BaseException as e:receipt['error']=repr(e);raise
 finally:receipt['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(output,receipt)
if __name__=='__main__':main()
