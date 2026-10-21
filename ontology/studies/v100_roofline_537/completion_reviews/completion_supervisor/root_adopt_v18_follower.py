from pathlib import Path
import json,hashlib,os,signal,time,subprocess,fcntl,datetime
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');P=Path('/tmp/roofline-completion-20261021');D=R/'ontology/studies/v100_roofline_537/completion_reviews/completion_supervisor'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return{'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],'children':(p/'task'/str(pid)/'children').read_text().strip(),'command_sha256':sha(p/'cmdline')}
old=json.loads((D/'root_adoption_command.json').read_text());q=proc(1576045);assert q['uid']==os.getuid()==10384 and q['start_ticks']=='78171838'and q['command']==old['command']and not q['children'];assert sha(R/'tools/v100_roofline_completion.py')=='72e07149a0e1d722e70fff8f61d925c139d3344d95c513888c4f8efdf4d1276d'
follower=json.loads((P/'columnar062_profiler2025/root_adoption.json').read_text());f=proc(follower['pid']);assert f['uid']==os.getuid()and f['start_ticks']==follower['start_ticks']and f['command']==follower['command'];reviews=json.loads((D/'controller_reviews_v17.json').read_text());key=str(f['pid'])+':'+f['start_ticks'];assert key not in reviews['controllers'];reviews['controllers'][key]={'command_sha256':f['command_sha256'],'source_reviewed_GPU_controller':True,'reason':'Rootandindependentreviewed one-shot062 new2025methodcolumnarKAJACCrecovery; wait whole021generation, skipqualified, maximum1collectorinvocation, all5checks/GPU6guards/explicitadmittedmethod retained, noadditionalmaster/rounds','source_sha256':follower['controller_sha256'],'method_review_sha256':follower['method_review_sha256']};reviewpath=D/'controller_reviews_v18.json';assert not reviewpath.exists();save(reviewpath,reviews)
cp=json.loads((RAW/'completion_20261021_v17/checkpoint.json').read_text());assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds'];assert not(RAW/'completion_20261021_v18').exists();handle=os.pidfd_open(q['pid']);signal.pidfd_send_signal(handle,signal.SIGSTOP)
try:
 for _ in range(100):
  live=proc(q['pid'])
  if live['state']=='T':break
  time.sleep(.01)
 assert live['state']=='T'and live['start_ticks']==q['start_ticks']and not live['children']and live['command']==q['command'];cp=json.loads((RAW/'completion_20261021_v17/checkpoint.json').read_text());assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds']
except BaseException:signal.pidfd_send_signal(handle,signal.SIGCONT);raise
for n in ['root_adoption_command.json','root_monitoring_frontier.json']:
 t=D/(Path(n).stem+'_v17.json');assert not t.exists();t.write_bytes((D/n).read_bytes())
save(D/'supervisor_v17_superseded.json',{'time':now(),'process':q,'checkpoint':cp,'reason':'Add one exactrootadopted bounded062follower; preserveall41previousIDs, sourcehash/publishingauthority/twoexistingrounds unchanged','GPU_signals':0})
signal.pidfd_send_signal(handle,signal.SIGTERM);signal.pidfd_send_signal(handle,signal.SIGCONT)
for _ in range(100):
 try:live=proc(q['pid'])
 except FileNotFoundError:break
 assert live['start_ticks']==q['start_ticks']
 if live['state']=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Old supervisor stilllive; no replacement')
os.close(handle)
with(RAW/'finite_completion_supervisor.lock').open('a')as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
argv=list(old['command']);argv[argv.index('--output-root')+1]=str(RAW/'completion_20261021_v18');argv[argv.index('--controller-reviews')+1]=str(reviewpath);argv+=['--controller',key];assert argv.count('--controller')==42
log=RAW/'completion_20261021_v18_stdout.jsonl'
with log.open('xb')as out:child=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 p=RAW/'completion_20261021_v18/checkpoint.json'
 if p.exists():break
 if child.poll()is not None:raise RuntimeError('Newsupervisor startupfailed; retainstdout, noautomaticrestart')
 time.sleep(.1)
new=proc(child.pid);assert new['uid']==os.getuid()and new['command']==argv and new['state']not in ['Z','X'];cp=json.loads(p.read_text());assert not cp['steps']and not cp['rounds'];config=json.loads((RAW/'completion_20261021_v18/configuration.json').read_text());assert config['publication_stratum']==old['publication_stratum']and config['max_retry_rounds']==2
receipt={**old,'time':now(),'pid':new['pid'],'start_ticks':new['start_ticks'],'command':argv,'stdout':str(log),'replaces':'v17 exactowned idle retiredusingpidfd; retain40historicalcontrollers+059060and062explicitboundedfollowers=42; samefinite039041','controller_count':42,'new_follower':follower};save(D/'root_adoption_command.json',receipt);save(D/'root_monitoring_frontier.json',{'time':now(),'supervisor':new,'controller_count':42,'command':argv,'warning':'Nohistoricalrestart; calibrationpass/newmethodadoption/queuedrecovery are notbenchmarkcapture success.'});(D/'root_adopt_v18_follower.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'supervisor':new['pid'],'start_ticks':new['start_ticks'],'controllers':42,'source_sha256':sha(R/'tools/v100_roofline_completion.py'),'bounded_follower':key}))
