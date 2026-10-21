from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time,shutil
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');D=R/'ontology/studies/v100_roofline_537/completion_reviews/completion_supervisor';P=Path('/tmp/roofline-completion-20261021');F=P/'uncovered_custom_followups_056_058'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],'children':(p/'task'/str(pid)/'children').read_text()}
review=json.loads((F/'launch_review.json').read_text());assert sha(F/'launch_review.json')=='e4661abad8506e2183b17f5c4db5953a0c7da440b2bdeb520c9e8c57ed1074aa'
assert sha(F/'wait_then_collect_three.py')==review['controller_sha256']
for p,h in review['readonly_bindings'].items():assert sha(p)==h,p
inverse=json.loads((F/'root_inverse_plan_review.json').read_text());assert len(inverse)==5 and all(not x['residual_fields']and x['oldbindings_preserved']for x in inverse)
assert shutil.disk_usage('/tmp').free>3*(1<<30)
root_review={'time':utc(),'root_launch_admissible':True,'scope':'Exactly three sequential one-shot operational collectors for five previously failed slots, old2022method only, GPU6only/sharedlock/originalguards/fullchecks/deadlines/selectors unchanged','launch_review_sha256':sha(F/'launch_review.json'),'controller_sha256':review['controller_sha256'],'readonly_binding_checks':len(review['readonly_bindings']),'inverse_plan_review_sha256':sha(F/'root_inverse_plan_review.json'),'GPU_build_tests_by_review':0,'no_new_method_adoption':True,'qualifications':['One retry is not an assertion of a cure for rc255/guard stops.','Inherited GPU idle waiting is outside execution deadlines; no total finish forecast.','ASSERT historical three-model review title remains, narrowed actual reviewed models are CUDAjl/JACC only.','All prior histories retained; only fresh outputs, retained auxiliary policy, no foreign signals.']};dump(F/'root_launch_adoption_review.json',root_review)
old=json.loads((D/'root_adoption_command.json').read_text());pid=old['pid'];q=proc(pid);assert q['uid']==os.getuid()and q['start_ticks']==old['start_ticks']and q['command']==old['command']and not q['children']
assert 'completion_20261021_v12'in old['command'][old['command'].index('--output-root')+1]
assert not (RAW/'completion_20261021_v13').exists()
os.kill(pid,signal.SIGSTOP);time.sleep(.15)
try:
 q=proc(pid);c=json.loads((RAW/'completion_20261021_v12/checkpoint.json').read_text());assert q['state']=='T'and not q['children']and c['status']=='waiting_explicit_owned_controllers'and not c['steps']and not c['rounds']
except BaseException:os.kill(pid,signal.SIGCONT);raise
for name in ['root_adoption_command','root_monitoring_frontier']:shutil.copyfile(D/(name+'.json'),D/(name+'_v12.json'))
dump(RAW/'completion_20261021_v12/root_superseded_while_waiting.json',{'time':utc(),'pid':pid,'start_ticks':q['start_ticks'],'reason':'Add five exact source-equivalent one-shot recoveries056/057/058, nochildren/noexecutedsteps/retries','no_children':True,'no_steps_or_rounds':True,'only_owned_waiting_supervisor_signaled':True})
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
for _ in range(100):
 try:
  x=proc(pid)
  if x['start_ticks']!=q['start_ticks']or x['state']=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise RuntimeError('old owned supervisor did not quiesce')
argv=json.loads((F/'future_argv.json').read_text())['argv']
with (RAW/'followups_056_058_controller_stdout.jsonl').open('xb')as out:follower=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.2);f=proc(follower.pid);assert f['state']not in ['Z','X'];key=str(f['pid'])+':'+f['start_ticks'];custom=json.loads((D/'controller_reviews_v12.json').read_text());custom['controllers'][key]={'command_sha256':hashlib.sha256((Path('/proc')/str(f['pid'])/'cmdline').read_bytes()).hexdigest(),'source_reviewed_GPU_controller':True,'reason':'Root source/binding/inverseplan reviewed three sequential one-shot oldmethod collectors for five customheldslots; exactparents/flags/fullchecks/GPU6guard preserved','review':str(F/'root_launch_adoption_review.json'),'review_sha256':sha(F/'root_launch_adoption_review.json'),'controller_sha256':review['controller_sha256']};dump(D/'controller_reviews_v13.json',custom)
cmd=[v.replace('completion_20261021_v12','completion_20261021_v13').replace('controller_reviews_v12.json','controller_reviews_v13.json')for v in old['command']]+['--controller',key]
with (RAW/'completion_20261021_v13_stdout.jsonl').open('xb')as out:child=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 checkpoint=RAW/'completion_20261021_v13/checkpoint.json'
 if checkpoint.exists():break
 time.sleep(.1)
else:raise RuntimeError('Newv13 failed startup; root recovery required, preserve exactcontrollers/no repeat')
now=proc(child.pid);nc=json.loads(checkpoint.read_text());assert now['state']not in ['Z','X']and nc['status']=='waiting_explicit_owned_controllers'and not nc['steps']and not nc['rounds']
receipt={**old,'time':utc(),'pid':now['pid'],'start_ticks':now['start_ticks'],'command':cmd,'stdout':str(RAW/'completion_20261021_v13_stdout.jsonl'),'controller_count':old['controller_count']+1,'replaces':'v12waiting/nochildren/noexecutedstepsorretries','custom056_058_follower':f,'custom056_058_root_review':{'path':str(F/'root_launch_adoption_review.json'),'sha256':sha(F/'root_launch_adoption_review.json')}}
dump(D/'root_adoption_command.json',receipt);dump(D/'root_monitoring_frontier.json',{'time':utc(),'supervisor':{'pid':now['pid'],'start_ticks':now['start_ticks']},'controller_count':receipt['controller_count'],'command':cmd,'warning':'Historical identities preserved; do not restart superseded supervisors/collectors orfailedv11'})
a=R/'.porting-state/portable_julia/active_work_537.json';state=json.loads(a.read_text());state['roofline_campaign']['completion_supervisor_v13']=receipt;state['roofline_campaign']['custom056_058_follower']=f;front=state['current_roofline_frontier'];front['updated_at']=utc();front['previously_uncovered_custom_recovery_slots']=front.pop('uncovered_custom_recovery_slots',[])
for slot in front['previously_uncovered_custom_recovery_slots']:slot['state']='Root-reviewed one-shot follower queued after exact original parent; not a passed capture'
front['custom056_058_follower']=f;state['updated_at']=utc();dump(a,state)
shutil.copyfile(Path(__file__),D/'root_adopt_v13.py');print(json.dumps({'supervisor_pid':now['pid'],'start_ticks':now['start_ticks'],'follower':f,'controller_count':receipt['controller_count']}))
