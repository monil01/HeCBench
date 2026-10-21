#!/usr/bin/env python3
"""Finite allreduce-only twoGPU proposal. NOT authorized merely by file existence.
Without --execute only fresh CPU directories/plans are staged; no subprocesses.
Execution requires exact proposal/runner SHA plus root review and explicit user
GPU5+6 authorization receipt. This file has NOT been executed or tested.
"""
import argparse,array,csv,datetime,fcntl,hashlib,importlib.util,json,os,re,resource,secrets,signal,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
MODELS=('cuda','cuda_jl','ka','jacc')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):
 temp=path.with_suffix(path.suffix+'.new');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def verify(bound):
 for path,h in bound.items():
  if not Path(path).is_file()or sha(path)!=h:raise RuntimeError('readonlybindingchanged '+path)
def expand(x,slot):
 if isinstance(x,str):return x.replace('{slot}',str(slot)).replace('{scratch}',str(slot/'scratch'))
 if isinstance(x,list):return [expand(y,slot)for y in x]
 if isinstance(x,dict):return {expand(k,slot):expand(y,slot)for k,y in x.items()}
 return x
def stat_pid(pid):
 p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'uid':p.stat().st_uid,'sid':os.getsid(pid),'starttime':f[19],'state':f[0],'exe':os.readlink(p/'exe')}
def own_members(sid):
 out={}
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    z=stat_pid(int(p.name))
    if z['uid']==os.getuid()and z['sid']==sid and z['state']!='Z':out[int(p.name)]=z
   except(OSError,ValueError):pass
 return out
def kill_own_session(sid,sig):
 # Never signal by process name, untrusted registry, arbitrary PID, or another SID.
 for pid,before in own_members(sid).items():
  try:
   after=stat_pid(pid)
   if(after['uid'],after['sid'],after['starttime'])!=(os.getuid(),sid,before['starttime']):raise RuntimeError('ownsessionidentitychanged')
   os.kill(pid,sig)
  except ProcessLookupError:pass

def device_snapshot(bundle):
 cmd=['nvidia-smi','-i','5,6','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader,nounits']
 q=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=10);devices={}
 for row in csv.reader(q.stdout.splitlines()):
  if len(row)!=4:raise RuntimeError('GPUidentityqueryschema')
  i,uuid,pci,name=[v.strip()for v in row];devices[i]={'uuid':uuid,'pci':pci,'name':name}
 if {k:v['uuid']for k,v in devices.items()}!=bundle['physical_GPU_UUIDs']:raise RuntimeError('bothphysicalUUIDsrequired')
 p=subprocess.run(['nvidia-smi','-i','5,6','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10);apps=[]
 for row in csv.reader(p.stdout.splitlines()):
  if len(row)!=3 or row[0].strip()not in bundle['physical_GPU_UUIDs'].values():raise RuntimeError('unrecognizedGPUprocessquery')
  apps.append({'gpu_uuid':row[0].strip(),'pid':int(row[1].strip()),'name':row[2].strip()})
 return {'observed_at':utc(),'devices':devices,'compute_processes':apps}
def registry_records(registry,token,sid):
 records=[]
 for name in ['rank0.json','rank1.json','profiler.json']:
  p=registry/name
  if not p.exists():continue
  try:r=json.loads(p.read_text())
  except json.JSONDecodeError:continue
  if p.is_symlink()or p.stat().st_uid!=os.getuid()or r.get('token')!=token or r.get('uid')!=os.getuid()or r.get('sid')!=sid:raise RuntimeError('authenticatedregistrymismatch '+name)
  if name.startswith('rank')and r.get('rank')!=int(name[4]):raise RuntimeError('rankregistryrole')
  records.append(r)
 return records
def assert_activity(snapshot,registry,token,sid,plan):
 records=registry_records(registry,token,sid);allowed={}
 for r in records:
  try:live=stat_pid(r['pid'])
  except(FileNotFoundError,ProcessLookupError):continue
  if (live['uid'],live['sid'],live['starttime'])!=(r['uid'],sid,r['starttime']):raise RuntimeError('registeredPIDidentitychanged')
  expected=[]
  if r.get('role')=='profiler':expected=[str(Path(plan['ncu_executable']).resolve())]
  else:
   exe=plan['rank_target_argv'][0];expected=[str(Path(exe).resolve())]
   if 'julia'in exe:expected.append('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia')
  if live['exe']in expected:allowed[r['pid']]=r
 # No same-session wildcard or startup CUDA grace: real workers register BEFORE
 # exec/CUDA; profiler registers before NCU exec. A launcher that forks fails.
 for app in snapshot['compute_processes']:
  r=allowed.get(app['pid'])
  if r is None:raise RuntimeError('foreign_or_unregistered_GPU_PID '+json.dumps(app))
  if r.get('role')=='profiler'and app['gpu_uuid']!=snapshot['devices']['6']['uuid']:raise RuntimeError('profilerunexpectedGPU5context')
 return records

def storage_check(slot,plan):
 payload=0;aux=0
 for p in slot.rglob('*'):
  if p.is_symlink():raise RuntimeError('unexpectedwritablesymlink '+str(p))
  if not p.is_file():continue
  size=p.stat().st_size
  if p.parent==slot/'scratch'and p.name in[f'trial{i}.bin'for i in range(1,6)]:
   if size>plan['payload_bytes_per_trial']:raise RuntimeError('payloadperfilecap')
   payload+=size
  else:aux+=size
 if payload>plan['payload_all5_bytes']or aux>plan['auxiliary_budget_bytes']:raise RuntimeError('perattemptstoragebudget')
 if shutil.disk_usage(slot).free<plan['reserve_bytes']:raise RuntimeError('freespacereserve')
 return {'payload_bytes':payload,'auxiliary_bytes':aux,'free_bytes':shutil.disk_usage(slot).free}
def child_limits():resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
def run_job(cmd,slot,env,timeout,bundle,plan,role,token):
 log=slot/(role+'.log');guard=bundle['write_guard']['path'];actual=[sys.executable,guard,'--allow-dir',str(slot),'--',*cmd]
 began=time.monotonic();result={'role':role,'command':actual,'started_at':utc(),'GPU_monitor_snapshots':[]};maps=set();registry=slot/'scratch/registry';last=0;process=None;stop=None
 try:
  with log.open('wb')as out:
   process=subprocess.Popen(actual,cwd=slot/'scratch',env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_limits)
   sid=process.pid;leader=stat_pid(sid)
   while process.poll()is None:
    try:
     if time.monotonic()-began>timeout:raise RuntimeError('observer_wall_timeout')
     storage_check(slot,plan)
     if time.monotonic()-last>=1:
      last=time.monotonic();snap=device_snapshot(bundle)
      if role.startswith('build'):
       if snap['compute_processes']:raise RuntimeError('GPUactivityduringbuild')
       records=[]
      else:
       records=assert_activity(snap,registry,token,sid,plan)
       if time.monotonic()-began>plan['MPI_registry_startup_deadline_seconds']and len([r for r in records if 'rank'in r and 'role'not in r])!=2:raise RuntimeError('exact2rankregistrystartuptimeout')
      result['GPU_monitor_snapshots'].append(snap)
      for r in records:
       try:
        for l in(Path('/proc')/str(r['pid'])/'maps').read_text().splitlines():
         f=l.split()[-1]
         if f.startswith('/')and '.so'in f and Path(f).is_file():maps.add(f)
       except OSError:pass
    except Exception as error:stop=str(error);break
    time.sleep(.25)
   if stop:
    kill_own_session(sid,signal.SIGTERM);end=time.monotonic()+10
    while own_members(sid)and time.monotonic()<end:time.sleep(.2)
    kill_own_session(sid,signal.SIGKILL)
   try:code=process.wait(timeout=10)
   except subprocess.TimeoutExpired:kill_own_session(sid,signal.SIGKILL);code=process.wait(timeout=10)
   end=time.monotonic()+10
   while own_members(sid)and time.monotonic()<end:time.sleep(.2)
   if own_members(sid):stop=stop or'own_session_outlived_leader';kill_own_session(sid,signal.SIGKILL)
   end=time.monotonic()+10
   while own_members(sid)and time.monotonic()<end:time.sleep(.2)
   if own_members(sid):raise RuntimeError('ownsessionnotquiescent_noartifactacceptance')
   result.update(returncode=code,stop_reason=stop,session=sid,own_session_quiescent=True,leader_identity=leader)
 except BaseException:
  if process is not None:kill_own_session(process.pid,signal.SIGTERM);time.sleep(.2);kill_own_session(process.pid,signal.SIGKILL)
  raise
 finally:
  result.update(finished_at=utc(),wall_seconds=time.monotonic()-began,log_sha256=sha(log)if log.exists()else None,observed_runtime_DSO_bindings={p:sha(p)for p in sorted(maps)if Path(p).is_file()});dump(slot/(role+'_provenance.json'),result)
 return result

def check_counter_rows(slot,plan,snapshot,rank1):
 path=slot/'counters.csv'
 with path.open(newline='')as f:rows=list(csv.reader(f))
 index=next((i for i,row in enumerate(rows)if 'Metric Name'in row and'Kernel Name'in row),None)
 if index is None:raise RuntimeError('NCUrawCSVheaderrequired')
 header=rows[index];groups={}
 for values in rows[index+1:]:
  if len(values)!=len(header):continue
  r=dict(zip(header,values))
  if not r.get('Metric Name'):continue
  key=tuple(r.get(x,'')for x in ['ID','Process ID','Context','Stream','Kernel Name']);g=groups.setdefault(key,{'identity':{x:r.get(x)for x in ['ID','Process ID','Context','Stream','Kernel Name','Device','Grid Size','Block Size']},'metrics':{}})
  if r['Metric Name']in g['metrics']:raise RuntimeError('duplicatemetric')
  g['metrics'][r['Metric Name']]=float(r['Metric Value'].replace(',',''))
 if len(groups)!=1:raise RuntimeError('exactoneisolatedaddcountergrouprequired')
 pci=snapshot['devices']['6']['pci'];domain,bus,dev=pci.split(':');expected=(int(domain,16),int(bus,16),int(dev.split('.')[0],16));out=[]
 for g in groups.values():
  ident=g['identity'];name=ident['Kernel Name']
  if not re.search(plan['kernel_regex'],name)or int(ident['Process ID'])!=rank1:raise RuntimeError('counterrank1/familymismatch')
  actual=tuple(g['metrics'].get('device__attribute_pci_'+key+'_id')for key in ['domain','bus','device'])
  if actual!=expected:raise RuntimeError('rawcounterPCI_notphysicalGPU6')
  for key,want in [('Grid Size',[32,1,1]),('Block Size',[256,1,1])]:
   values=[int(v)for v in re.findall(r'\d+',ident[key]or'')]
   if values!=want:raise RuntimeError('exact32x256geometryrequired')
  if g['metrics'].get('gpu__time_duration.sum',0)<=0:raise RuntimeError('positivecountersamplerequired')
  out.append(g)
 return {'counter_sha256':sha(path),'groups':out,'physical_GPU6_UUID':snapshot['devices']['6']['uuid'],'physical_GPU6_PCI':expected,'rank1_worker_pid':rank1,'scope':'One retainedtrial2 size1M firstrepeat rank1 privateadd only; not collective/wholeROI.'}

def validate_outputs(slot,plan,token):
 log=(slot/'profile.log').read_text(errors='replace');trials=[];begins=[];ends=[];done=[]
 for line in log.splitlines():
  for label,out in [('STUDY_TRIAL ',trials),('ALLREDUCE_PROFILE_BEGIN ',begins),('ALLREDUCE_PROFILE_END ',ends),('ALLREDUCE_PROFILE_DONE ',done)]:
   if label in line:
    raw=line.split(label,1)[1];obj,end=json.JSONDecoder().raw_decode(raw);out.append(obj)
 if sorted(x['trial']for x in trials)!=[1,2,3,4,5]or any(x.get('correct')is not True or x.get('checked_values')!=5269603 for x in trials):raise RuntimeError('fivefullcorrecttrialsrequired')
 if len(begins)!=1 or len(ends)!=1 or len(done)!=2:raise RuntimeError('exactmarkersbothranksrequired')
 registry=slot/'scratch/registry';rs={}
 for rank in(0,1):
  r=json.loads((registry/f'rank{rank}.json').read_text())
  if r['rank']!=rank or r['token']!=token:raise RuntimeError('finalrankregistry')
  rs[rank]=r
 begin=begins[0];end=ends[0]
 if any(begin.get(k)!=v for k,v in {'rank':1,'trial':2,'n':1048576,'repeat':1,'offset':0,'count':524288,'pid':rs[1]['pid']}.items()):raise RuntimeError('exactwindowbegin')
 if any(end.get(k)!=v for k,v in {'rank':1,'trial':2,'n':1048576,'repeat':1,'authored_add_dispatches':1,'pid':rs[1]['pid']}.items()):raise RuntimeError('exactwindowend')
 if sorted(x['rank']for x in done)!=[0,1]:raise RuntimeError('bothrankfinalguards')
 for x in done:
  rank=x['rank']
  if(x['pid'],x['ring_add_calls'],x['candidates'],x['enters'],x['exits'])!=(rs[rank]['pid'],96,3 if rank else 0,1 if rank else 0,1 if rank else 0):raise RuntimeError('fullcontroller96/finalbranchguards')
 spec=importlib.util.spec_from_file_location('allreduce_fullchecker',HERE/'check_artifact.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);artifacts=[]
 for trial in trials:
  t=trial['trial'];decl=trial['output_artifact'];path=slot/'scratch'/f'trial{t}.bin'
  if Path(decl['path']).name!=path.name or decl['dtype']!='float32'or decl['count']!=5269603:raise RuntimeError('originalartifactcontract')
  artifacts.append(checker.check(path))
 return {'trials':trials,'begin':begin,'end':end,'done':done,'artifacts':artifacts,'rank_registry':rs}

def stage(bundle,out):
 if out.exists()or out.is_symlink()or not out.resolve().is_relative_to('/tmp')or out.resolve()==Path('/tmp'):raise RuntimeError('freshstrict/tmpoutputrequired')
 out.mkdir(mode=0o700,parents=True)
 for m in MODELS:
  slot=out/'allreduce'/m;scratch=slot/'scratch'
  for name in ['profile_outputs','tmp','cuda_cache','xdg_cache','julia_depot','mpi_tmp','ucx_posix','registry']:(scratch/name).mkdir(mode=0o700,parents=True)
  plan=expand(bundle['models'][m],slot);dump(slot/'plan.json',plan)
 dump(out/'staging_receipt.json',{'staged_at':utc(),'models':list(MODELS),'CPU_staging_only':True,'runtime_asset_copy_bytes':0,'GPU_runs':0,'build_runs':0})
def require_authorization(a,bundle,bundle_sha):
 if not a.authorization or not a.authorization_sha256:raise RuntimeError('explicitrootreview+userGPU5authorizationreceiptrequired')
 if sha(a.authorization)!=a.authorization_sha256:raise RuntimeError('authorizationreceiptSHA')
 r=json.loads(a.authorization.read_text())
 expected={'execution_authorized':True,'authorization_origin':'explicit_user_approval_recorded_by_root','proposal_sha256':bundle_sha,'runner_sha256':sha(Path(__file__)),'physical_GPU_UUIDs':bundle['physical_GPU_UUIDs'],'models':list(MODELS),'deadline_seconds_per_model':1800}
 if any(r.get(k)!=v for k,v in expected.items())or not r.get('user_authorization_reference')or not r.get('root_independent_review_reference'):raise RuntimeError('additionalGPU5userauthorization/rootreviewnotestablished')
 return r

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proposal',type=Path,default=HERE/'proposal.json');p.add_argument('--output-root',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256');a=p.parse_args();os.umask(0o077);proposal=a.proposal.resolve();bundle=json.loads(proposal.read_text());bundle_sha=sha(proposal)
 if bundle['benchmark']!='allreduce'or bundle['status']!='READY_FOR_ROOT_REVIEW_NOT_EXECUTION_AUTHORIZED':raise RuntimeError('exactallreduceproposalrequired')
 verify(bundle['all_readonly_bindings']);authorization=require_authorization(a,bundle,bundle_sha)if a.execute else None
 out=a.output_root.resolve();stage(bundle,out)
 if not a.execute:print(json.dumps({'status':'CPU_staged_not_executed','output_root':str(out),'requires_additional_user_GPU5_authorization':True}));return
 fds=[]
 try:
  # Finite lock acquisition; a sharedGPU6 lock does not itself authorize GPU5.
  until=time.monotonic()+60
  for path in [bundle['common_lock'],bundle['pair_lock']]:
   fd=os.open(path,os.O_RDWR|os.O_CREAT,0o600);fds.append(fd)
   while True:
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:
     if time.monotonic()>until:raise RuntimeError('finiteownedlockwaitexpired')
     time.sleep(.25)
  verify(bundle['all_readonly_bindings'])
  for m in MODELS:
   slot=out/'allreduce'/m;plan=json.loads((slot/'plan.json').read_text());token=secrets.token_hex(32);env=os.environ.copy()
   if any(env.get(k)for k in ['LD_PRELOAD','CUDA_INJECTION64_PATH','NVBIT_TOOL']):raise RuntimeError('unreviewedambientinjection')
   env.update(plan['environment_overrides']);env.update(STUDY_PROFILE_REGISTRY=str(slot/'scratch/registry'),STUDY_PROFILE_TOKEN=token)
   required=plan['payload_all5_bytes']+plan['auxiliary_budget_bytes']+plan['reserve_bytes']
   if shutil.disk_usage(slot).free<required:raise RuntimeError('wholefivepayloadauxreservepreflight')
   snap=device_snapshot(bundle)
   if snap['compute_processes']:raise RuntimeError('bothGPUsmustbeidle_defer')
   env['STUDY_EXPECTED_GPU6_PCI']=snap['devices']['6']['pci'];prov={'started_at':utc(),'model':m,'status':'started','proposal_sha256':bundle_sha,'authorization_receipt_sha256':a.authorization_sha256,'authorization':authorization,'device_before':snap,'staging_plan_sha256':sha(slot/'plan.json'),'source_bound_before':True,'fullchecks_required':5,'window_scope':'retainedtrial2size1Mfirstrepeat_rank1privateaddonly','primary_timing_replacement':False}
   try:
    began=time.monotonic();builds=[]
    for i,command in enumerate(plan['build_commands']):
     r=run_job(command,slot,env,min(plan['build_deadline_seconds'],max(1,plan['observer_deadline_seconds']-(time.monotonic()-began))),bundle,plan,'build'+str(i),token);builds.append(r)
     if r['returncode']!=0 or r['stop_reason']:raise RuntimeError('isolatedbuildfailed')
    prov['builds']=builds;verify(bundle['all_readonly_bindings']);snap=device_snapshot(bundle)
    if snap['compute_processes']:raise RuntimeError('bothGPUsmustbeidleafterbuild')
    env['STUDY_EXPECTED_GPU6_PCI']=snap['devices']['6']['pci'];prov['device_before_profile']=snap
    targets={str(p):sha(p)for p in(slot/'scratch').iterdir()if p.is_file()};prov['fresh_build_output_bindings']=targets
    # Keep exact accepted launcher/HPCX environment and 2ranks. Launcher appends
    # accepted MPI command; only per-rank dispatch placement is added.
    command=[sys.executable,plan['session_entry'],sys.executable,plan['accepted_MPI_launcher'],sys.executable,plan['rank_dispatch'],str(slot/'plan.json')]
    budget=plan['observer_deadline_seconds']-(time.monotonic()-began)
    if budget<=0:raise RuntimeError('observerdeadlineexhaustedbeforeprofile')
    result=run_job(command,slot,env,budget,bundle,plan,'profile',token);prov['profile']=result
    if result['returncode']!=0 or result['stop_reason']:raise RuntimeError('profileprocessfailed')
    full=validate_outputs(slot,plan,token);prov['full_correctness']=full
    prov['counter_identity']=check_counter_rows(slot,plan,snap,full['rank_registry'][1]['pid']);verify(bundle['all_readonly_bindings']);verify(targets)
    after=device_snapshot(bundle)
    if after['compute_processes']:raise RuntimeError('bothGPUsnotidleafterownedquiescence')
    prov.update(status='profile_completed_all5checks_and_GPU6identity_verified',device_after=after,source_bound_after=True,cache_policy='all_cold_private_kernel_replay',replay_mode='kernel',communication_replayed=False)
   except BaseException as error:prov.update(status='failed_no_roofline_acceptance',error=repr(error));raise
   finally:
    prov['finished_at']=utc();dump(slot/'profile_provenance.json',prov)
    # Preserve all outputs/logs/rawcounters/registry; no cleanup/deletion routine.
 except BaseException as error:
  dump(out/'controller_failure.json',{'time':utc(),'error':repr(error),'primary_data_deleted':False});raise
 finally:
  for fd in reversed(fds):fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
if __name__=='__main__':main()
