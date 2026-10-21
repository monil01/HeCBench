from pathlib import Path
import json,hashlib,copy,difflib,fcntl,os,datetime
R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');O=Path(__file__).resolve().parent;P55=O.parent/'rbc055_guard_recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def replace(v,a,b):
 if isinstance(v,str):return v.replace(str(a),str(b))
 if isinstance(v,list):return[replace(x,a,b)for x in v]
 if isinstance(v,dict):return{replace(k,a,b):replace(x,a,b)for k,x in v.items()}
 return v
def proc(pid,start):
 p=Path('/proc')/str(pid)
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split();a=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x]
  if st[19]!=start:return {'pid':pid,'start_ticks':start,'already_finished_or_reused':True}
  if p.stat().st_uid!=os.getuid():raise RuntimeError('Original controller ownership mismatch')
  return {'pid':pid,'start_ticks':start,'uid':p.stat().st_uid,'argv':a,'identity':str(pid)+':'+start,'state_at_review':st[0]}
 except FileNotFoundError:return {'pid':pid,'start_ticks':start,'already_finished_or_reused':True}
frozen=O/'frozen_collect_32a9.py'
if frozen.exists() and sha(frozen)!='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c':raise RuntimeError('Frozen private source drift')
if not frozen.exists():
 frozen.write_bytes((P55/'frozen_collect_32a9.py').read_bytes());frozen.chmod(0o444)
assert sha(frozen)=='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
pins=json.loads((P55/'frozen_method_review.json').read_text())['execution_bindings'];pins={k:v for k,v in pins.items()if not k.endswith('frozen_collect_32a9.py')};pins[str(frozen)]=sha(frozen)
for p,s in pins.items():assert sha(p)==s
cases=[{'benchmark':'assert','models':['cuda_jl','jacc'],'old':'batch045-assert-isolated-phases','new':'batch056-assert-isolated-recovery','review_key':'isolated_assertion_review','flag':'--isolated-assertion-reviewed','timeout':900,'parent':proc(1269351,'76747416')},{'benchmark':'atomicSystemWide','models':['cuda_jl','jacc'],'old':'batch043-atomic-app-replay','new':'batch057-atomic-app-recovery','review_key':'application_replay_review','flag':'--application-replay-reviewed','timeout':1800,'parent':proc(1269349,'76747395')},{'benchmark':'columnarSolver','models':['cuda_jl'],'old':'batch021-retry-columnarSolver','new':'batch058-columnar-cudajl-recovery','review_key':None,'flag':None,'timeout':600,'parent':proc(1141998,'76117190')}]
all_bindings={};reports=[];commands=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for case in cases:
  b=case['benchmark'];oldroot=RAW/case['old'];newroot=RAW/case['new']
  if newroot.exists() and (not (O/'unexecuted_staging_history/v1/failure.json').exists() or list(newroot.rglob('profile_provenance.json'))):raise RuntimeError('Only own preserved unexecuted metadata staging may resume')
  plans=[];review=None;oldreviewpath=None;newreviewpath=O/(b+'_method_review.json')
  for m in case['models']:
   oldp=oldroot/b/m/'plan.json';old=json.loads(oldp.read_text());new=replace(copy.deepcopy(old),oldroot,newroot);slot=newroot/b/m;scratch=slot/'scratch'
   for sub in ['profile_outputs','ncu_tmp','profile_source']:(scratch/sub).mkdir(parents=True,exist_ok=True)
   if case['review_key']:
    link=old[case['review_key']];rp=Path(link['path']);assert sha(rp)==link['sha256']
    if review is None:
     oldreviewpath=rp;rd=json.loads(rp.read_text());review=replace(copy.deepcopy(rd),oldroot,newroot);review['models']={k:v for k,v in review['models'].items()if k in [b+'/'+x for x in case['models']]}
    new=replace(new,rp,newreviewpath)
   changes=[];recovered=[]
   for path,h in old['execution_bindings'].items():
    source=Path(path)
    if path.startswith(str(oldroot)+'/'):
     target=Path(str(source).replace(str(oldroot),str(newroot)))
     if source.exists():bs=source.read_bytes()
     elif b=='atomicSystemWide':
      source=O.parent/'remaining_replay/atomic_application_pass.py';bs=source.read_bytes();recovered.append({'old_deleted_scratch_path':path,'recovered_from':meta(source),'expected_unchanged_sha256':h,'reason':'Old collector cleaned scratch after retaining receipt; exact canonical helper with same recorded SHA remains. No algorithm reconstruction.'})
     else:raise RuntimeError('Missing source binding '+path)
     assert hashlib.sha256(bs).hexdigest()==h
     if target.suffix=='.json':
      val=replace(json.loads(bs),oldroot,newroot);dump(target,val)
     else:
      assert str(oldroot).encode()not in bs;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bs)
     new['execution_bindings'][str(target)]=sha(target);changes.append({'old':path,'new':str(target),'before_sha256':h,'after_sha256':sha(target),'scope':'metadata path relocation only'if target.suffix=='.json'else'byteidentical helper'})
    else:
     assert source.is_file()and sha(source)==h,('Original readonly binding drift',path)
   # Update ASSERT spec hash and review records; C/JL/math/checker sources unchanged.
   if b=='assert':
    rec=review['models'][b+'/'+m];spec=slot/'phase_spec.json';rec['phase_spec']['sha256']=sha(spec);rec['execution_bindings'][str(spec)]=sha(spec)
    extra=(new.get('review')or{}).get('entry',{}).get('additional_current_readonly_input_bindings',{})
    if str(spec)in extra:extra[str(spec)]=sha(spec)
   if case['review_key']:
    rec=review['models'][b+'/'+m]
    for ch in changes:rec['execution_bindings'][ch['new']]=ch['after_sha256']
    rec['execution_bindings'].update(pins)
    new['execution_bindings'].update(pins)
   else:new['execution_bindings'].update(pins)
   plans.append((oldp,old,new,slot,changes,recovered))
  if review is not None:dump(newreviewpath,review)
  if b=='columnarSolver':
   oldsel=oldroot/'bounded_selector.json';selector=newroot/'bounded_selector.json';selector.write_bytes(oldsel.read_bytes());assert sha(selector)==sha(oldsel)
  for oldp,old,new,slot,changes,recovered in plans:
   if review is not None:
    for k in ['review',case['review_key']]:
     if isinstance(new.get(k),dict)and new[k].get('path')==str(newreviewpath):new[k]['sha256']=sha(newreviewpath)
    new['execution_bindings'][str(newreviewpath)]=sha(newreviewpath)
    extra=(new.get('review')or{}).get('entry',{}).get('additional_current_readonly_input_bindings',{})
    if str(newreviewpath)in extra:extra[str(newreviewpath)]=sha(newreviewpath)
   if b=='columnarSolver':new['execution_bindings'][str(selector)]=sha(selector)
   pp=slot/'plan.json';dump(pp,new)
   for path,h in new['execution_bindings'].items():assert sha(path)==h,('Fresh binding mismatch',path)
   if review is not None:
    assert all(new['execution_bindings'].get(p)==h for p,h in review['models'][b+'/'+new['model']]['execution_bindings'].items())
   result=Path(new['reference_result']);assert sha(result)==new['reference_result_sha256']and sha(result.with_name('config.json'))==new['raw_config_sha256']
   reports.append({'benchmark':b,'model':new['model'],'old_plan':meta(oldp),'new_plan':meta(pp),'old_review':meta(oldreviewpath)if oldreviewpath else None,'new_review':meta(newreviewpath)if review else None,'path_only_changes':changes,'exact_helper_recovery':recovered,'selected_result':meta(result),'selected_config':meta(result.with_name('config.json')),'source_math_and_all_checks_unchanged':True,'diff':''.join(difflib.unified_diff(json.dumps(old,indent=2).splitlines(True),json.dumps(new,indent=2).splitlines(True),fromfile=str(oldp),tofile=str(pp))),'historical_failure_provenance':meta(oldp.parent/'profile_provenance.json')})
   all_bindings[str(pp)]=sha(pp);all_bindings.update(new['execution_bindings']);all_bindings[str(oldp)]=sha(oldp);all_bindings[str(oldp.parent/'profile_provenance.json')]=sha(oldp.parent/'profile_provenance.json')
  argv=['python3',str(frozen),'--source-root',str(R),'--plan-root',str(newroot),'--timeout',str(case['timeout']),'--retain-auxiliary','--skip-qualified-selected']
  if case['flag']:argv.append(case['flag'])
  if b=='columnarSolver':argv+=['--selection-overrides',str(selector)]
  commands.append({'benchmark':b,'plan_root':str(newroot),'models':case['models'],'argv':argv,'wait_original_controller':case['parent'],'max_collector_invocations':1,'deadline_seconds_per_model':case['timeout'],'all5_full_checks_required':True,'replay_scope':'positive assertion kernel replay + five isolated pre/post contexts'if b=='assert'else'application/grid/strict replay; fullCPU/GPUproperty check perpass'if b=='atomicSystemWide'else'kernel replay, same bounded global first-specialization selector; full150000x1024 and all5 states/guards'})
for path,h in all_bindings.items():assert sha(path)==h
out={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PRIVATE_READY_FOR_ROOT_REVIEW_NOT_LAUNCHED','cases':commands,'plan_reviews':reports,'readonly_bindings':all_bindings,'frozen_collector':meta(frozen),'old2022_method_only':True,'old2025_calibration_adoption_presumed':False,'unchanged_old_NCU_wrapper':{'path':'/usr/lib/nsight-compute/ncu','sha256':pins['/usr/lib/nsight-compute/ncu']},'scope':'Exactly three sequential one-shot collectors for five selected failed slots; no retry loops, oldfailedraw preserved. Source/fixture/math/checks unchanged; fresh paths/reviews/pins only. Root permission/adoption required.','GPU_runs':0,'builds':0,'tests':0,'target_imports':0,'launches':0,'shared_edits':False}
dump(O/'staging_review.json',out);print(json.dumps({'cases':len(commands),'slots':len(reports),'readonly_bindings':len(all_bindings),'staging_review':meta(O/'staging_review.json')}))
