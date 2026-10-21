from pathlib import Path
import json,hashlib,os,signal,time,subprocess,fcntl,datetime,stat
R=Path('/home/imo/HeCBench_julia');P=Path('/tmp/roofline-completion-20261021');RAW=Path('/tmp/hecbench-roofline-v100-20261020');S=R/'ontology/studies/v100_roofline_537';D=S/'completion_reviews/completion_supervisor';OUT=D/'method_v5_presentation_v3_v16';STRATUM='bb70165f0fb0b0236774f801c42dd584f264d2f20b8b7b3ffe72193502cc77ef'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(d,indent=2)+'\n').encode();t=p.with_name(p.name+'.next');t.write_bytes(b);os.replace(t,p)
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return{'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],'children':(p/'task'/str(pid)/'children').read_text().strip(),'command_sha256':sha(p/'cmdline')}
def expect(pid,start,command=None):
 x=proc(pid);assert x['uid']==os.getuid()==10384 and x['start_ticks']==start and not x['children'] and x['state']not in ['Z','X'];assert command is None or x['command']==command;return x
old=json.loads((D/'root_adoption_command.json').read_text());sup=expect(1476003,'77711211',old['command']);cpu=expect(1471189,'77682370');assert cpu['command_sha256']=='ccf15abb2d3a666e96569d2fbdfbe7cba30c76fd5476213834eb320735c49313'
cp=json.loads((RAW/'completion_20261021_v15/checkpoint.json').read_text());assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds'];assert not OUT.exists()and not(RAW/'completion_20261021_v16').exists();assert sha(R/'tools/v100_roofline_completion.py')=='c26bfe8436993422529276796cbc8df6f2840b25f43552ccbf6e98ddce3bb74c'
files=[]
for x in json.loads((P/'profiler2025_method_patch/v5/authoring_bindings.json').read_text()):files.append((x['file'],x['base_sha256'],P/'profiler2025_method_patch/v5/proposed'/x['file'],x['v5_sha256']))
for x in json.loads((P/'profiler_method_presentation_v3/source_inventory.json').read_text()):
 if x['changed']:files.append((x['path'],x['base_sha256'],P/'profiler_method_presentation_v3/proposed'/x['path'],x['proposed_sha256']))
files.append(('tools/v100_roofline_completion.py','c26bfe8436993422529276796cbc8df6f2840b25f43552ccbf6e98ddce3bb74c',P/'completion_v16_v2_tool_bindings/proposed/tools/v100_roofline_completion.py','72e07149a0e1d722e70fff8f61d925c139d3344d95c513888c4f8efdf4d1276d'))
for rel,base,candidate,newsha in files:
 assert sha(candidate)==newsha,rel
 assert sha(R/rel)==base if base else not(R/rel).exists(),rel
method=S/'profiler_methods/ncu2025_2_0_V100_sm70_raw_v1';assert sha(method/'review.json')=='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca';assert sha(method/'method_bundle.json')=='12ad8f0b5d87fa45f14cd6546f2f3302307cb2f25f1b24bf8023eb919783e685'
OUT.mkdir();save(OUT/'preflight.json',{'time':now(),'supervisor':sup,'CPUrefresh':cpu,'checkpoint':cp,'files':[{'path':r,'base_sha256':b,'candidate':str(c),'adopted_sha256':h}for r,b,c,h in files],'publication_stratum':STRATUM,'selection_scope':'Explicit recorded2022 version diagnostic stratum; method identity/review remain unknown, no retrocertification. New2025 separate.','method_review_sha256':sha(method/'review.json'),'method_bundle_sha256':sha(method/'method_bundle.json')})
# Generation handles prevent signaling a PID reused after the immediate checks.
handles={x['pid']:os.pidfd_open(x['pid'])for x in [cpu,sup]};paused=[]
try:
 for x in [cpu,sup]:
  expect(x['pid'],x['start_ticks'],x['command']);signal.pidfd_send_signal(handles[x['pid']],signal.SIGSTOP);paused.append(x)
 for x in paused:
  for _ in range(100):
   q=proc(x['pid'])
   if q['state']=='T':break
   time.sleep(.01)
  assert q['state']=='T'and q['start_ticks']==x['start_ticks']and q['command']==x['command']and not q['children']
 cp=json.loads((RAW/'completion_20261021_v15/checkpoint.json').read_text());assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds'];save(OUT/'paused_recheck.json',{'time':now(),'processes':[proc(x['pid'])for x in paused],'checkpoint':cp})
except BaseException:
 for x in paused:signal.pidfd_send_signal(handles[x['pid']],signal.SIGCONT)
 raise
for name in ['root_adoption_command.json','root_monitoring_frontier.json']:
 target=D/(Path(name).stem+'_v15.json');assert not target.exists();target.write_bytes((D/name).read_bytes())
for x in [cpu,sup]:
 signal.pidfd_send_signal(handles[x['pid']],signal.SIGTERM);signal.pidfd_send_signal(handles[x['pid']],signal.SIGCONT)
 for _ in range(100):
  try:q=proc(x['pid'])
  except FileNotFoundError:break
  assert q['start_ticks']==x['start_ticks']
  if q['state']=='Z':break
  time.sleep(.1)
 else:raise RuntimeError('Exact idle parent did not exit; no tool replacement or new supervisor')
 os.close(handles[x['pid']])
with(RAW/'finite_completion_supervisor.lock').open('a')as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
save(OUT/'retired_idle_parents.json',{'time':now(),'processes':[cpu,sup],'GPU_signals':0,'reason':'Exact owned idle CPU parents only; nochildren/no stepsrounds; pidfd generation handles; singleton released.'})
marker=S/'.profiler_method_deployment_active.json'
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not marker.exists();save(marker,{'state':'installing_reviewed_sources','review':str(OUT/'preflight.json'),'old_sources':str(OUT/'old_sources'),'time':now()})
 for rel,base,candidate,newsha in files:
  target=R/rel;assert sha(target)==base if base else not target.exists(),rel
  if base:
   backup=OUT/'old_sources'/rel;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(target.read_bytes())
  mode=stat.S_IMODE(target.stat().st_mode)if target.exists()else 0o644;t=target.with_name(target.name+'.method_v16.next');t.write_bytes(candidate.read_bytes());os.chmod(t,mode);os.replace(t,target);assert sha(target)==newsha
 save(OUT/'sources_installed.json',{'time':now(),'files':{rel:sha(R/rel)for rel,_,_,_ in files},'raw_counter_math_parser_unchanged':sha(R/'tools/v100_roofline_classify.py'),'method_review_sha256':sha(method/'review.json'),'method_bundle_sha256':sha(method/'method_bundle.json')});marker.unlink()
argv=list(old['command']);argv[argv.index('--output-root')+1]=str(RAW/'completion_20261021_v16');argv+=['--publication-stratum',STRATUM];assert argv.count('--controller')==old['command'].count('--controller')==40
log=RAW/'completion_20261021_v16_stdout.jsonl'
with log.open('xb')as out:child=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 path=RAW/'completion_20261021_v16/checkpoint.json'
 if path.exists():break
 if child.poll()is not None:raise RuntimeError('New supervisor startup failed; inspect durable stdout, do not restart')
 time.sleep(.1)
new=proc(child.pid);assert new['command']==argv and new['state']not in ['Z','X'];newcp=json.loads(path.read_text());assert not newcp['steps']and not newcp['rounds']
config=json.loads((RAW/'completion_20261021_v16/configuration.json').read_text());assert config['publication_stratum']==STRATUM
receipt={**old,'time':now(),'pid':new['pid'],'start_ticks':new['start_ticks'],'command':argv,'stdout':str(log),'supervisor_source_sha256':sha(R/'tools/v100_roofline_completion.py'),'replaces':'v15 exact-owned idle CPU parents retired with pidfd; same40controllers and finite039041 rounds','publication_stratum':STRATUM,'method_adoption':str(OUT.relative_to(R)),'method_review_sha256':sha(method/'review.json'),'method_bundle_sha256':sha(method/'method_bundle.json')};save(D/'root_adoption_command.json',receipt);save(D/'root_monitoring_frontier.json',{'time':now(),'supervisor':new,'controller_count':40,'command':argv,'warning':'Historical controllers are not restartinstructions. New2025 separatelyadmitted; no newmethodbenchmarksuccess inferred.'})
# Explicitly reviewed replacement continuous CPU export command, same allowed monitor list.
refreshargv=list(cpu['command'])+['--publication-stratum',STRATUM];refreshlog=RAW/'refresh_v16_stdout.jsonl'
with refreshlog.open('xb')as out:refresh=subprocess.Popen(refreshargv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.3);fresh=proc(refresh.pid);assert fresh['state']not in ['Z','X']and fresh['command']==refreshargv;save(OUT/'new_CPU_refresh.json',{'time':now(),'process':fresh,'source_sha256':sha(R/'tools/v100_roofline_refresh.py'),'stdout':str(refreshlog),'publication_stratum':STRATUM,'scope':'CPU exports only, exact existing collector PID wait allowlist; v16 finalonce waits all40.'});save(OUT/'root_adoption.json',{'time':now(),'status':'reviewed_core_v5_presentation_v3_and_v16_deployed','supervisor':new,'CPUrefresh':fresh,'same_historical_controllers':40,'publication_stratum':STRATUM,'method_review_sha256':sha(method/'review.json'),'method_bundle_sha256':sha(method/'method_bundle.json'),'GPU_jobs_or_signals_launched_by_adoption':0,'scientific_campaign_complete':False});(OUT/'root_adopt_method_v16.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'status':'adopted','supervisor':new['pid'],'CPUrefresh':fresh['pid'],'publication_stratum':STRATUM,'method_review_sha256':sha(method/'review.json')}))
