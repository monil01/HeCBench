from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time,shutil
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');D=R/'ontology/studies/v100_roofline_537/completion_reviews/completion_supervisor';P=Path('/tmp/roofline-completion-20261021')
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],'children':(p/'task'/str(pid)/'children').read_text()}
old=json.loads((D/'root_adoption_command.json').read_text());pid=old['pid'];q=proc(pid);assert q['uid']==os.getuid()and q['start_ticks']==old['start_ticks']and q['command']==old['command']and q['state']=='T'and not q['children']
c=json.loads((RAW/'completion_20261021_v10/checkpoint.json').read_text());assert c['status']=='waiting_explicit_owned_controllers'and not c['steps']and not c['rounds']
f=proc(1429230);assert f['uid']==os.getuid()and not f['children']and f['command']==json.loads((P/'rbc055_guard_recovery/future_followup_argv.json').read_text())['argv'];key=str(f['pid'])+':'+f['start_ticks'];custom=json.loads((D/'controller_reviews_v11.json').read_text());assert custom['controllers'][key]['command_sha256']==hashlib.sha256((Path('/proc')/str(f['pid'])/'cmdline').read_bytes()).hexdigest()
assert not (RAW/'completion_20261021_v12').exists()
for name in ['root_adoption_command','root_monitoring_frontier']:shutil.copyfile(D/(name+'.json'),D/(name+'_v10.json'))
dump(RAW/'completion_20261021_v10/root_superseded_while_waiting.json',{'time':utc(),'pid':pid,'start_ticks':q['start_ticks'],'reason':'Adopt one reviewed RBC055 follower; attemptedv11 startup rejected singletonlock held by STOPpedv10; no v11steps/rounds/children ever existed','no_children':True,'no_steps_or_rounds':True,'only_owned_waiting_supervisor_signaled':True})
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
for _ in range(100):
 try:
  x=proc(pid)
  if x['start_ticks']!=q['start_ticks']or x['state']=='Z':break
 except FileNotFoundError:break
 time.sleep(.1)
else:raise RuntimeError('oldowned supervisor did not quiesce')
shutil.copyfile(D/'controller_reviews_v11.json',D/'controller_reviews_v12.json');cmd=[v.replace('completion_20261021_v10','completion_20261021_v12').replace('controller_reviews_v10.json','controller_reviews_v12.json')for v in old['command']]+['--controller',key]
with (RAW/'completion_20261021_v12_stdout.jsonl').open('xb')as out:child=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 checkpoint=RAW/'completion_20261021_v12/checkpoint.json'
 if checkpoint.exists():break
 time.sleep(.1)
else:raise RuntimeError('Newv12 supervisor failed startup; inspectpreservedstdout')
now=proc(child.pid);nc=json.loads(checkpoint.read_text());assert now['state']not in ['Z','X']and nc['status']=='waiting_explicit_owned_controllers'and not nc['steps']and not nc['rounds']
receipt={**old,'time':utc(),'pid':now['pid'],'start_ticks':now['start_ticks'],'command':cmd,'stdout':str(RAW/'completion_20261021_v12_stdout.jsonl'),'controller_count':old['controller_count']+1,'replaces':'v10waiting/nochildren/noexecutedstepsorretries; v11pretargetsingletonlockfailure preserved','rbc055_follower':f,'rbc055_independent_review':{'path':str(P/'rbc055_independent_review/independent_review.json'),'sha256':'f216a4048ea7ccf0c11de87f0ce9efeb06a0521431c7baf099830c84e1ebf2d6'}}
dump(D/'root_adoption_command.json',receipt);dump(D/'root_monitoring_frontier.json',{'time':utc(),'supervisor':{'pid':now['pid'],'start_ticks':now['start_ticks']},'controller_count':receipt['controller_count'],'command':cmd,'warning':'Historical identities preserved; do not restart superseded supervisors/collectors orfailedv11'})
a=R/'.porting-state/portable_julia/active_work_537.json';state=json.loads(a.read_text());state['roofline_campaign']['completion_supervisor_v12']=receipt;state['roofline_campaign']['rbc055_follower']=f;state['current_roofline_frontier']['supervisor_adoption']=str((D/'root_adoption_command.json').relative_to(R));state['updated_at']=utc();dump(a,state)
shutil.copyfile(P/'v11_adoption_failure.json',D/'v11_adoption_failure.json');shutil.copyfile(P/'root_adopt_v11.py',D/'root_adopt_v11.py');shutil.copyfile(Path(__file__),D/'root_recover_supervisor_v12.py')
print(json.dumps({'supervisor':now,'follower':f,'controller_count':receipt['controller_count']}))
