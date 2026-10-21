from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time,shutil
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');S=R/'ontology/studies/v100_roofline_537';D=S/'completion_reviews/completion_supervisor';P=Path('/tmp/roofline-completion-20261021');F=P/'rbc055_guard_recovery'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':(p/'cmdline').read_bytes().split(b'\0')[:-1],'children':(p/'task'/str(pid)/'children').read_text()}
# Root must explicitly provide an independently reviewed immutable receipt.
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--independent-review',type=Path,required=True);ap.add_argument('--independent-review-sha256',required=True);a=ap.parse_args()
assert sha(a.independent_review)==a.independent_review_sha256
ind=json.loads(a.independent_review.read_text());assert ind['root_launch_admissible']is True
review=json.loads((F/'launch_review.json').read_text());assert sha(F/'launch_review.json')=='d5486461b2ca6e6ea5d0e0fb25c3b8b7fce0ca0817152441718d6ebe89cb80db'
assert sha(F/'wait_then_collect.py')==review['follower_sha256']
for p,h in review['readonly_bindings'].items():assert sha(Path(p))==h,p
assert str(Path(shutil.which('ncu')).resolve())==review['NCU_wrapper']['path']
old=json.loads((D/'root_adoption_command.json').read_text());pid=old['pid'];before=proc(pid)
assert before['uid']==os.getuid()and before['start_ticks']==old['start_ticks']and not before['children']
assert [v.decode()for v in before['command']]==old['command']
assert 'completion_20261021_v10'in old['command'][old['command'].index('--output-root')+1]
assert not (RAW/'completion_20261021_v11').exists()
os.kill(pid,signal.SIGSTOP)
terminated=False
child=None
manager=None
try:
 time.sleep(.15);q=proc(pid);c=json.loads((RAW/'completion_20261021_v10/checkpoint.json').read_text())
 assert q['state']=='T'and q['command']==before['command']and q['start_ticks']==before['start_ticks']and not q['children']and c['status']=='waiting_explicit_owned_controllers'and not c['steps']and not c['rounds']
 argv=json.loads((F/'future_followup_argv.json').read_text())['argv']
 with (RAW/'followups_055_controller_stdout.jsonl').open('xb')as out:child=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.25);new=proc(child.pid);assert new['state']not in ['Z','X'];key=str(new['pid'])+':'+new['start_ticks']
 custom=json.loads((D/'controller_reviews_v10.json').read_text());custom['controllers'][key]={'command_sha256':hashlib.sha256((Path('/proc')/str(new['pid'])/'cmdline').read_bytes()).hexdigest(),'source_reviewed_GPU_controller':True,'reason':'Root and independent source review: exactly one frozen old-method RBC055 collector after exact050 generation ends, all full5checks/6000steps/retainedwindow/3600unchanged, GPU6 guard/lock preserved, skipsqualified.','review':str(a.independent_review),'review_sha256':a.independent_review_sha256,'follower_sha256':review['follower_sha256']};dump(D/'controller_reviews_v11.json',custom)
 cmd=[v.replace('completion_20261021_v10','completion_20261021_v11').replace('controller_reviews_v10.json','controller_reviews_v11.json')for v in old['command']]+['--controller',key]
 with (RAW/'completion_20261021_v11_stdout.jsonl').open('xb')as out:manager=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.5);now=proc(manager.pid);assert now['state']not in ['Z','X']
 checkpoint=RAW/'completion_20261021_v11/checkpoint.json'
 for _ in range(30):
  if checkpoint.exists():break
  time.sleep(.2)
 assert checkpoint.exists(),'new supervisor failed to checkpoint'
 nc=json.loads(checkpoint.read_text());assert nc['status']=='waiting_explicit_owned_controllers'and not nc['steps']and not nc['rounds']
 for name in ['root_adoption_command','root_monitoring_frontier']:shutil.copyfile(D/(name+'.json'),D/(name+'_v10.json'))
 dump(RAW/'completion_20261021_v10/root_superseded_while_waiting.json',{'time':utc(),'pid':pid,'start_ticks':q['start_ticks'],'reason':'Add finite independently reviewed RBC055 after guard interruption; v10 has nochildren/steps/rounds','no_children':True,'no_steps_or_rounds':True,'only_owned_waiting_supervisor_signaled':True})
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);terminated=True
 new['command']=[v.decode()for v in new['command']]
 receipt={**old,'time':utc(),'pid':now['pid'],'start_ticks':now['start_ticks'],'command':cmd,'stdout':str(RAW/'completion_20261021_v11_stdout.jsonl'),'controller_count':old['controller_count']+1,'replaces':'v10waiting/nochildren/noexecutedstepsorretries','rbc055_follower':new,'rbc055_independent_review':{'path':str(a.independent_review),'sha256':a.independent_review_sha256}}
 dump(D/'root_adoption_command.json',receipt);dump(D/'root_monitoring_frontier.json',{'time':utc(),'supervisor':{'pid':now['pid'],'start_ticks':now['start_ticks']},'controller_count':receipt['controller_count'],'command':cmd,'warning':'Historical identities preserved; do not restart superseded supervisors/collectors'})
 astate=R/'.porting-state/portable_julia/active_work_537.json';state=json.loads(astate.read_text());state['roofline_campaign']['completion_supervisor_v11']=receipt;state['roofline_campaign']['rbc055_follower']=new;state['updated_at']=utc();dump(astate,state)
 print(json.dumps({'time':utc(),'rbc055_follower':new,'supervisor_v11_pid':now['pid'],'start_ticks':now['start_ticks']}))
except BaseException as error:
 # Before any new controller exists, restoring the exact idle supervisor is safe.
 # Once a follower exists, keep old supervisor stopped until root repairs its
 # explicit frontier; never silently run two finite retry supervisors.
 dump(P/'v11_adoption_failure.json',{'time':utc(),'error':repr(error),'old_pid':pid,'old_start_ticks':before['start_ticks'],'old_terminated':terminated,'new_follower_pid':child.pid if child else None,'new_supervisor_pid':manager.pid if manager else None,'requires_root_recovery':child is not None})
 if not terminated and child is None:os.kill(pid,signal.SIGCONT)
 raise
