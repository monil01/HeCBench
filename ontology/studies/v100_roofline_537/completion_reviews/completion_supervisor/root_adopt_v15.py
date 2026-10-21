from pathlib import Path
import json,hashlib,os,signal,time,subprocess,shutil,datetime
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');D=R/'ontology/studies/v100_roofline_537/completion_reviews/completion_supervisor';P=Path('/tmp/roofline-completion-20261021')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],'children':(p/'task'/str(pid)/'children').read_text().strip()}
old=json.loads((D/'root_adoption_command.json').read_text());q=proc(old['pid']);assert q['pid']==1468993 and q['start_ticks']=='77670277' and q['uid']==os.getuid() and q['command']==old['command'] and not q['children']
cp=json.loads((RAW/'completion_20261021_v14/checkpoint.json').read_text());assert cp['status']=='waiting_explicit_owned_controllers' and not cp['steps'] and not cp['rounds']
src=R/'tools/v100_roofline_completion.py';assert sha(src)=='b223403d39842726436d8c8c0818381ce147955e4aa965d05f19811d254b1b4b'
# Apply the reviewed two-hunk forwarding patch to an isolated copy first.
prep=P/'completion_display_trait_passthrough/root_v15';(prep/'tools').mkdir(parents=True,exist_ok=True);shutil.copyfile(src,prep/'tools'/src.name)
subprocess.run(['patch','-p1','--input',str(P/'completion_display_trait_passthrough/minimal.patch')],cwd=prep,check=True,capture_output=True)
candidate=prep/'tools'/src.name;assert sha(candidate)=='f133bd5a037fa2fd23ca8da6ba68dc23d075cfc9b973927c79b47bbaf438f4f7'
s=candidate.read_text();needle="'timeout':a.timeout,'freeze_every':a.freeze_every,'tool_bindings':self.bindings,'existing_round_states':a.existing_round_state,"
assert s.count(needle)==1;s=s.replace(needle,needle+"\n   'current537_display_review_sha256':a.current537_display_review_sha256,");candidate.write_text(s)
authority=old['current537_display_authority'];assert sha(R/authority['path'])==authority['sha256']
assert not (RAW/'completion_20261021_v15').exists()
for name in ['root_adoption_command.json','root_monitoring_frontier.json']:
 target=D/(Path(name).stem+'_v14.json');assert not target.exists();shutil.copyfile(D/name,target)
# Recheck exact idle identity immediately before terminating only this root-owned supervisor.
assert proc(q['pid'])==q
os.kill(q['pid'],signal.SIGTERM);os.kill(q['pid'],signal.SIGCONT)
for _ in range(100):
 try:
  live=proc(q['pid'])
  if live['state']=='Z':break
  assert live['start_ticks']==q['start_ticks']
 except FileNotFoundError:break
 time.sleep(.1)
else:raise RuntimeError('Old supervisor did not release; no replacement launched')
save(D/'supervisor_v14_superseded.json',{'time':now(),'process':q,'checkpoint':cp,'reason':'CPU final refresh forwards and resume-binds exact display authority; zero GPU signals'})
shutil.copyfile(candidate,src)
argv=list(old['command']);argv[argv.index('--output-root')+1]=str(RAW/'completion_20261021_v15');argv+=['--current537-display-review-sha256',authority['sha256']]
log=RAW/'completion_20261021_v15_stdout.jsonl'
with log.open('xb') as out:c=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.5);new=proc(c.pid);assert new['state'] not in ['Z','X'] and new['uid']==os.getuid() and new['command']==argv
newcp=json.loads((RAW/'completion_20261021_v15/checkpoint.json').read_text());assert not newcp['steps'] and not newcp['rounds']
receipt={**old,'time':now(),'pid':new['pid'],'start_ticks':new['start_ticks'],'command':argv,'stdout':str(log),'supervisor_source_sha256':sha(src),'replaces':'v14 exact-owned idle/nochildren/noexecutedsteps; released singleton before fresh v15','current537_display_authority':authority}
save(D/'root_adoption_command.json',receipt);save(D/'root_monitoring_frontier.json',{'time':now(),'supervisor':new,'controller_count':40,'command':argv,'warning':'Historical controllers are evidence, not restart instructions; calibration003 queued, no new method admitted'})
a=R/'.porting-state/portable_julia/active_work_537.json';state=json.loads(a.read_text());state['roofline_campaign']['completion_supervisor_v15']=receipt;state['current_roofline_frontier'].update(updated_at=now(),current_supervisor_version='v15',current_supervisor=new,display_export_audit=str(P/'display_traits_actual_export_audit.json'));state['updated_at']=now();save(a,state)
shutil.copyfile(P/'display_traits_actual_export_audit.json',R/'ontology/studies/v100_roofline_537/completion_reviews/current537_display_traits/actual_export_audit.json');shutil.copyfile(Path(__file__),D/'root_adopt_v15.py');save(D/'v15_source_review.json',{'time':now(),'base_sha256':old['supervisor_source_sha256'],'worker_patch_sha256':sha(P/'completion_display_trait_passthrough/minimal.patch'),'adopted_source_sha256':sha(src),'root_addition':'Bind forwarded display authority SHA in immutable configuration for exact resume comparison','scope':'CPU final refresh only; same 40 historical controller identities and finite rounds; no GPU jobs launched'})
print(json.dumps({'supervisor':new,'source_sha256':sha(src),'controllers':40}))
