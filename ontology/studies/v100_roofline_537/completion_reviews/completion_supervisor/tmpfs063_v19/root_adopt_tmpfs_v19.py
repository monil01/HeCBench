from pathlib import Path
import json,hashlib,os,signal,time,subprocess,fcntl,stat,datetime
R=Path('/home/imo/HeCBench_julia');P=Path('/tmp/roofline-completion-20261021');RAW=Path('/tmp/hecbench-roofline-v100-20261020');S=R/'ontology/studies/v100_roofline_537';D=S/'completion_reviews/completion_supervisor';O=D/'tmpfs063_v19';FINAL=P/'tmpfs063_final_root_review'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def proc(pid):
 p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':f[19],'state':f[0],'command':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],'children':(p/'task'/str(pid)/'children').read_text().strip(),'command_sha256':sha(p/'cmdline')}
def expect(record):
 q=proc(record['pid']);assert q['uid']==os.getuid()==10384 and q['start_ticks']==record['start_ticks']and q['command']==record['command']and q['state']not in ['Z','X'];return q
old=load(D/'root_adoption_command.json');assert old['pid']==1597201 and old['start_ticks']=='78275440'and old['controller_count']==42;sup=expect(old);cpu_record=load(D/'method_v5_presentation_v3_v16/new_CPU_refresh.json');cpu=expect(cpu_record['process']);assert cpu['pid']==1570693 and cpu['start_ticks']=='78145087';assert not sup['children']and not cpu['children'];cp=load(RAW/'completion_20261021_v18/checkpoint.json');assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds']
inputs=load(FINAL/'deployment_inputs.json');files=inputs['files'];assert len(files)==6;assert sha(FINAL/'storage_review.json')==inputs['storage_review_sha256']=='47bc4ad70a7182ae11c98f2c65d3cf55792079a4774cd6a18f87ccdeb7d0e0dd';assert sha(FINAL/'launch_review.json')==inputs['launch_review_sha256']=='3f0364022879dbbc79d4d725df0b322c528acdde44b287e3bc583f63e5c6cdb8';assert sha(inputs['root_scientific_review']['path'])==inputs['root_scientific_review']['sha256']
for x in files:
 assert sha(x['candidate'])==x['adopted_sha256'];target=R/x['path'];assert sha(target)==x['base_sha256']if x['base_sha256']else not target.exists()
assert sha(R/'tools/v100_roofline_collect.py')=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116';assert not O.exists()and not(RAW/'completion_20261021_v19').exists();O.mkdir();save(O/'preflight.json',{'time':now(),'supervisor':sup,'CPUrefresh':cpu,'checkpoint':cp,'deployment_inputs':inputs,'GPU_jobs_or_signals_by_preflight':0})
handles={x['pid']:os.pidfd_open(x['pid'])for x in [cpu,sup]};paused=[]
try:
 for x in [cpu,sup]:
  q=expect(x);assert not q['children'];signal.pidfd_send_signal(handles[x['pid']],signal.SIGSTOP);paused.append(x)
 for x in paused:
  for _ in range(100):
   q=expect(x)
   if q['state']=='T':break
   time.sleep(.01)
  assert q['state']=='T'and not q['children']
 cp=load(RAW/'completion_20261021_v18/checkpoint.json');assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds'];save(O/'paused_recheck.json',{'time':now(),'processes':[proc(x['pid'])for x in paused],'checkpoint':cp})
except BaseException:
 for x in paused:signal.pidfd_send_signal(handles[x['pid']],signal.SIGCONT)
 raise
for n in ['root_adoption_command.json','root_monitoring_frontier.json']:
 target=D/(Path(n).stem+'_v18.json');assert not target.exists();target.write_bytes((D/n).read_bytes())
for x in [cpu,sup]:
 signal.pidfd_send_signal(handles[x['pid']],signal.SIGTERM);signal.pidfd_send_signal(handles[x['pid']],signal.SIGCONT)
 for _ in range(100):
  try:q=proc(x['pid'])
  except FileNotFoundError:break
  assert q['start_ticks']==x['start_ticks']
  if q['state']=='Z':break
  time.sleep(.1)
 else:raise RuntimeError('ExactidleCPUparent has not exited; no source install/new processes')
 os.close(handles[x['pid']])
with(RAW/'finite_completion_supervisor.lock').open('a')as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
save(O/'retired_idle_CPU_parents.json',{'time':now(),'processes':[cpu,sup],'GPU_signals':0,'reason':'OnlyexactownedidleCPUgenerations/nochildren/noexecutedstepsrounds retired withpidfd; singletonreleased.'})
marker=S/'.tmpfs_v19_deployment_active.json'
with Path('/tmp/hecbench-roofline-metadata.lock').open()as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not marker.exists();save(marker,{'time':now(),'status':'installing_reviewed_sources','preflight':str(O/'preflight.json')})
 for x in files:
  target=R/x['path'];oldsha=x['base_sha256'];assert sha(target)==oldsha if oldsha else not target.exists()
  if oldsha:
   backup=O/'old_sources'/x['path'];backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(target.read_bytes())
  mode=stat.S_IMODE(target.stat().st_mode)if target.exists()else 0o644;tmp=target.with_name(target.name+'.tmpfs_v19.next');tmp.write_bytes(Path(x['candidate']).read_bytes());os.chmod(tmp,mode);os.replace(tmp,target);assert sha(target)==x['adopted_sha256']
 for name in ['source_scientific_review.json','storage_review.json','launch_review.json','deployment_inputs.json']:(O/name).write_bytes((FINAL/name).read_bytes())
 save(O/'sources_installed.json',{'time':now(),'sources':{x['path']:sha(R/x['path'])for x in files},'canonical_C21_unchanged':sha(R/'tools/v100_roofline_collect.py'),'parser_unchanged':sha(R/'tools/v100_roofline_classify.py'),'guard_unchanged':sha(R/'tools/v100_profile_write_guard.py')});marker.unlink()
# Launch one source-reviewed follower. Ordinary UID10384 only. It waits all exact
# predecessor generations and may later create only reviewed0700 /dev/shm scratch.
controller=P/'su3_tmpfs063_native_v2/wait_then_collect.py';assert sha(controller)==inputs['controller_sha256']=='7dd2fe76eef7204bd1841731153d65698fc11e9c8c3be1816ab2bf818c81ba9f'
argv=['/usr/bin/python3',str(controller),'--execute','--launch-review',str(FINAL/'launch_review.json'),'--launch-review-sha256',inputs['launch_review_sha256'],'--tmpfs-review',str(FINAL/'storage_review.json'),'--tmpfs-review-sha256',inputs['storage_review_sha256']]
with(P/'su3_tmpfs063_native_v2/root_controller_stdout.log').open('xb')as out:follower=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.3);f=proc(follower.pid);assert f['uid']==os.getuid()and f['state']not in ['Z','X']and f['command']==argv;follower_receipt={'time':now(),'process':f,'command':argv,'controller_sha256':sha(controller),'maximum_collector_invocations':1,'storage_review_sha256':inputs['storage_review_sha256'],'launch_review_sha256':inputs['launch_review_sha256'],'scope':'ReviewedunprivilegedGPU6onlystoragefallback; noGPU5, no primarytimingreplacement, no primarydata deletion. Queued isnotverification.'};save(O/'follower_root_adoption.json',follower_receipt)
reviews=load(D/'controller_reviews_v18.json');key=str(f['pid'])+':'+f['start_ticks'];assert key not in reviews['controllers'];reviews['controllers'][key]={'command_sha256':f['command_sha256'],'source_reviewed_GPU_controller':True,'source_sha256':sha(controller),'reason':'Exactrootandindependentreviewed bounded063SU3tmpfsfollower; wholeoriginal+046049wait, skipqualified, max1collectorcall/default2022/all5checks/leafguard/memoryfloors. Noadditionalround/master.'};reviewpath=D/'controller_reviews_v19.json';assert not reviewpath.exists();save(reviewpath,reviews)
command=list(old['command']);command[command.index('--output-root')+1]=str(RAW/'completion_20261021_v19');command[command.index('--controller-reviews')+1]=str(reviewpath);command+=['--controller',key];assert command.count('--controller')==43
log=RAW/'completion_20261021_v19_stdout.jsonl'
with log.open('xb')as out:child=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 path=RAW/'completion_20261021_v19/checkpoint.json'
 if path.exists():break
 if child.poll()is not None:raise RuntimeError('v19startupfailed; retainlogs, noautomaticrestart')
 time.sleep(.1)
new=proc(child.pid);assert new['state']not in ['Z','X']and new['command']==command;cp=load(path);assert not cp['steps']and not cp['rounds'];cfg=load(RAW/'completion_20261021_v19/configuration.json');assert cfg['publication_stratum']==old['publication_stratum']and cfg['max_retry_rounds']==2
# Replace only the exact original CPU refresh command, retaining its allowed PIDs.
refreshlog=RAW/'refresh_v19_stdout.jsonl'
with refreshlog.open('xb')as out:refresh=subprocess.Popen(cpu['command'],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.3);fresh=proc(refresh.pid);assert fresh['state']not in ['Z','X']and fresh['command']==cpu['command'];save(O/'new_CPU_refresh.json',{'time':now(),'process':fresh,'source_sha256':sha(R/'tools/v100_roofline_refresh.py'),'stdout':str(refreshlog),'publication_stratum':old['publication_stratum']})
receipt={**old,'time':now(),'pid':new['pid'],'start_ticks':new['start_ticks'],'command':command,'stdout':str(log),'controller_count':43,'replaces':'v18andCPUrefresh exactownedidleparents retired;42previouscontrollerIDs retained plusbounded063; samefinite039041','supervisor_source_sha256':sha(R/'tools/v100_roofline_completion.py'),'legacy_controller_collector_source_sha256':old['collector_source_sha256'],'collector_source_sha256':sha(R/'tools/v100_roofline_collect.py'),'legacy_admission_tools':old['admission_tools'],'admission_tools':{n:sha(R/'tools'/n)for n in ['v100_roofline_summarize.py','v100_roofline_audit.py','v100_roofline_allreduce_admission.py']},'CPU_refresh_with_display_authority':fresh,'refresh_source_sha256':sha(R/'tools/v100_roofline_refresh.py'),'storage_adoption':str(O.relative_to(R)),'new_follower':follower_receipt};save(D/'root_adoption_command.json',receipt);save(D/'root_monitoring_frontier.json',{'time':now(),'supervisor':new,'controller_count':43,'command':command,'CPUrefresh':fresh,'warning':'Queuedfallback/sourceadmission are not successfulcapture/wholeROI/scientific completion. No historicalrestart.'});save(O/'root_adoption.json',{'time':now(),'status':'reviewed_storage_and_offline_admission_sources_v19_and063_deployed','supervisor':new,'CPUrefresh':fresh,'follower':f,'source_inputs':26,'historical_controllers_retained':42,'total_controllers':43,'GPUcollector_signals':0,'scientific_campaign_complete':False});(O/'root_adopt_tmpfs_v19.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'status':'deployed','supervisor':new['pid'],'supervisor_start_ticks':new['start_ticks'],'CPUrefresh':fresh['pid'],'controller063':f['pid'],'controller063_start_ticks':f['start_ticks'],'controllers':43,'pinned_core_presentation_source_inputs':26,'canonical_C21_unchanged':sha(R/'tools/v100_roofline_collect.py'),'GPU_signals':0}))
