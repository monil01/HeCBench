from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time,shutil,fcntl
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');S=R/'ontology/studies/v100_roofline_537';D=S/'completion_reviews/completion_supervisor';P=Path('/tmp/roofline-completion-20261021');C=P/'profiler2025_calibration003';T=P/'current537_display_refresh_passthrough/complete_combined'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return {'pid':pid,'uid':p.stat().st_uid,'start_ticks':v[19],'state':v[0],'command':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],'children':(p/'task'/str(pid)/'children').read_text(),'wchan':(p/'wchan').read_text()}
def stop_idle(expected):
 q=proc(expected['pid']);assert q['uid']==os.getuid()and q['start_ticks']==expected['start_ticks']and q['command']==expected['command']and not q['children']and q['wchan']=='hrtimer_nanosleep'
 os.kill(q['pid'],signal.SIGSTOP);time.sleep(.15);z=proc(q['pid'])
 if z['state']!='T'or z['children']or z['start_ticks']!=q['start_ticks']or z['command']!=q['command']:
  os.kill(q['pid'],signal.SIGCONT);raise RuntimeError('Owned monitor left safe idle boundary')
 return q
def end_stopped(q):
 z=proc(q['pid']);assert z['uid']==os.getuid()and z['start_ticks']==q['start_ticks']and z['command']==q['command']and z['state']=='T'and not z['children'];os.kill(q['pid'],signal.SIGTERM);os.kill(q['pid'],signal.SIGCONT)
 for _ in range(100):
  try:
   x=proc(q['pid'])
   if x['start_ticks']!=q['start_ticks']or x['state']=='Z':return
  except FileNotFoundError:return
  time.sleep(.1)
 raise RuntimeError('Owned monitor failed to quiesce')
cal=json.loads((C/'delivery.json').read_text());assert sha(C/'calibrate_controller.py')==cal['controller_sha256']=='b47be69ff2658a52f9d5a8c7def5d0a40671e3d5cd3e86f2fcaaea5fc23bbe4e';assert sha(C/'source_binding_manifest.json')==cal['manifest_sha256']=='6793c9f873f493865da7b0898552980990cc2521fb8e0255d78cc5d94edbadd2'
manifest=json.loads((C/'source_binding_manifest.json').read_text());assert len(manifest['bindings'])==83
for p,h in manifest['bindings'].items():assert sha(p)==h,p
assert not (RAW/'method_calibration_ncu2025_2_0_003').exists()and not (RAW/'completion_20261021_v14').exists()
old=json.loads((D/'root_adoption_command.json').read_text());assert old['pid']==1451257 and old['start_ticks']=='77590874'
refresh=proc(1215425);assert refresh['start_ticks']=='76480236'
oldsuper=stop_idle(old)
try:oldrefresh=stop_idle(refresh)
except BaseException:os.kill(oldsuper['pid'],signal.SIGCONT);raise
cp=json.loads((RAW/'completion_20261021_v13/checkpoint.json').read_text());assert cp['status']=='waiting_explicit_owned_controllers'and not cp['steps']and not cp['rounds']
for name in ['root_adoption_command','root_monitoring_frontier']:shutil.copyfile(D/(name+'.json'),D/(name+'_v13.json'))
dump(D/'cpu_refresh_v1_superseded.json',{'time':utc(),'process':oldrefresh,'no_children':True,'state_before_signal':'idle_interval_sleep','reason':'Forward explicit current537 display authority and inherited EX-lock lease'})
end_stopped(oldrefresh);end_stopped(oldsuper)
dump(RAW/'completion_20261021_v13/root_superseded_while_waiting.json',{'time':utc(),'process':oldsuper,'no_children':True,'no_steps_or_rounds':True,'reason':'Pin reviewed trait-enabled refresh and add fresh003 exactfrequency-unit validation; no method adoption'})
with open('/tmp/hecbench-roofline-metadata.lock')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 subprocess.run(['git','apply',str(T/'minimal.patch')],cwd=R,check=True)
 explorer=R/'ontology/tools/report/explorer.js';text=explorer.read_text();oldlink='../studies/julia_abstractions_447/domains/${encodeURIComponent(id)}.json';assert text.count(oldlink)==1;text=text.replace(oldlink,'../../${encodeURI(STUDY.study_path)}/domains/${encodeURIComponent(id)}.json');explorer.write_text(text)
 authority=json.loads((T/'root_review.template.json').read_text());assert sha(R/authority['loader']['path'])==authority['loader']['sha256']=='4d47981aca1f4ce13271e5c43f23ec1707d2db27ee5c797982300c05fd06b358';assert sha(R/authority['overlay']['path'])==authority['overlay']['sha256']=='9b0788ff1472540bb5b22a8a5c847d63dcd9e8801a6cfb029c7e07b95424ab6c';assert sha(R/authority['source_adoption']['path'])==authority['source_adoption']['sha256']
 authority.update(status='root_approved_current537_display_traits',reviewed_at=utc(),reviewer='root',scope='Seven source-bound display mapping/geometry corrections and14curated-domain aliases; no frozen source/classification/cohort/scoring mutation',limitations=['Display only; no numerical, invocation phase or measured bottleneck promotion.','Explicit authority SHA and borrowed canonical metadata EX-lock FD required for parent-locked exports.','Original graph scoring inputs retained; corrected display values remain separately labeled.'],additional_root_fix='Scientific record links use actual STUDY.study_path for537 and447; historical hardcoded447 link removed')
 apath=S/'completion_reviews/current537_display_traits/root_review.json';dump(apath,authority);authority_sha=sha(apath)
 graph=json.loads((R/'ontology/relationships/julia_relationships.json').read_text());dump(P/'display_traits_original_edges.json',graph['edges'])
calreview={'time':utc(),'source_reviewed':True,'runtime_validated':False,'controller':{'path':cal['controller'],'sha256':cal['controller_sha256']},'manifest':{'path':cal['manifest'],'sha256':cal['manifest_sha256']},'scope':'Fresh003 same2existingELFs/full9cases/fullhostchecks/27metrics/GPU6guard/sharedlock/600s; exactly3frequency unit aliases hz=cycle/second factor1. 002failed preserved; no profiler method adoption.','actual002_diagnostic':{'path':str(C/'actual002_unit_review.json'),'sha256':sha(C/'actual002_unit_review.json')},'OS_changes':0,'new_builds_by_preparation':0};dump(C/'root_source_review.json',calreview)
with (RAW/'method_calibration_ncu2025_2_0_003_stdout.jsonl').open('xb')as out:calchild=subprocess.Popen(cal['root_launch_argv'],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.2);cq=proc(calchild.pid);assert cq['state']not in ['Z','X'];key=str(cq['pid'])+':'+cq['start_ticks'];custom=json.loads((D/'controller_reviews_v13.json').read_text());custom['controllers'][key]={'command_sha256':hashlib.sha256((Path('/proc')/str(cq['pid'])/'cmdline').read_bytes()).hexdigest(),'source_reviewed_GPU_controller':True,'reason':calreview['scope'],'review':str(C/'root_source_review.json'),'review_sha256':sha(C/'root_source_review.json')};dump(D/'controller_reviews_v14.json',custom)
cmd=[v.replace('completion_20261021_v13','completion_20261021_v14').replace('controller_reviews_v13.json','controller_reviews_v14.json')for v in old['command']]+['--controller',key]
with (RAW/'completion_20261021_v14_stdout.jsonl').open('xb')as out:superchild=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 checkpoint=RAW/'completion_20261021_v14/checkpoint.json'
 if checkpoint.exists():break
 time.sleep(.1)
else:raise RuntimeError('v14startupfailed; inspectexactownedjobs andstdout, noautomaticrepeat')
sq=proc(superchild.pid);assert sq['state']not in ['Z','X'];ncp=json.loads(checkpoint.read_text());assert ncp['status']=='waiting_explicit_owned_controllers'and not ncp['steps']and not ncp['rounds']
rargv=oldrefresh['command']+['--wait-pid',str(sq['pid']),'--current537-display-review-sha256',authority_sha]
with (RAW/'refresh_traits_root_v14_stdout.jsonl').open('xb')as out:rchild=subprocess.Popen(rargv,cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(.2);rq=proc(rchild.pid);assert rq['state']not in ['Z','X']
receipt={**old,'time':utc(),'pid':sq['pid'],'start_ticks':sq['start_ticks'],'command':cmd,'stdout':str(RAW/'completion_20261021_v14_stdout.jsonl'),'controller_count':old['controller_count']+1,'replaces':'v13waiting/nochildren/noexecutedstepsorretries','calibration003_controller':cq,'current537_display_authority':{'path':str(apath.relative_to(R)),'sha256':authority_sha},'CPU_refresh_with_display_authority':rq,'refresh_source_sha256':sha(R/'tools/v100_roofline_refresh.py')}
dump(D/'root_adoption_command.json',receipt);dump(D/'root_monitoring_frontier.json',{'time':utc(),'supervisor':{'pid':sq['pid'],'start_ticks':sq['start_ticks']},'controller_count':receipt['controller_count'],'command':cmd,'warning':'Historical identities retained; do not restart superseded controllers/supervisors orfailed002/001'})
a=R/'.porting-state/portable_julia/active_work_537.json';state=json.loads(a.read_text());state['roofline_campaign']['completion_supervisor_v14']=receipt;state['roofline_campaign']['profiler_calibration003']=cq;state['roofline_campaign']['cpu_refresh_supervisor']=rq;state['current_roofline_frontier'].update(updated_at=utc(),current537_display_authority=receipt['current537_display_authority'],profiler_calibration002_outcome='failed firstcapture unitlabels; no newmethodadoption',profiler_calibration003_state='queued source-reviewed actual9case validation, no methodadoption');state['updated_at']=utc();dump(a,state)
shutil.copyfile(Path(__file__),D/'root_adopt_v14.py');print(json.dumps({'supervisor':{'pid':sq['pid'],'start_ticks':sq['start_ticks']},'calibration003':{'pid':cq['pid'],'start_ticks':cq['start_ticks']},'CPU_refresh':{'pid':rq['pid'],'start_ticks':rq['start_ticks']},'display_authority_sha256':authority_sha,'controller_count':receipt['controller_count']}))
