from pathlib import Path
import json,gzip,hashlib,copy,fcntl,datetime
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;RAW=Path('/tmp/hecbench-roofline-v100-20261020')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def info(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def load(p):p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def move(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[move(y,a,b)for y in x]
 if isinstance(x,dict):return{k:move(v,a,b)for k,v in x.items()}
 return x
jobs=[];proofs=[];sourceproof=[];selectors=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 for n,m in [('dispatch',m)for m in ['cuda','cuda_jl','ka','jacc']]+[('slu','ka')]:
  led=load(S/'benchmarks'/f'{n}.json');ct=led['accepted_named_contract'];st=led['models'][m];pid=st['latest_profile_attempt']['profile_id'];cap=W/pid;oldraw=gzip.decompress((cap/'plan.json.gz').read_bytes());p=json.loads(oldraw);pr=load(cap/'profile_provenance.json.gz');assert p['configuration_id']==ct['configuration_id']and p['raw_config_sha256']==ct['raw_config_sha256']and p['reference_result_sha256']==ct['result_sha256'];assert sha(p['reference_result'])==ct['result_sha256']and sha(Path(p['reference_result']).with_name('config.json'))==ct['raw_config_sha256'];assert pr.get('profiler_method')is None and'Version 2022.4.1.0'in pr['profiler_version'];assert sha(pr['profiler_executable'])==pr['profiler_executable_sha256'];assert pr['profile']['stop_reason']=='wall_timeout';assert pr['operational_deadline']['requested_seconds']==(600 if n=='dispatch'else 300);assert st['profiling_status']!='profile_process_completed';assert ('batch021-retry-dispatch'if n=='dispatch'else'batch018-slu-bounded')in pid
  root=RAW/('batch073'if n=='dispatch'else'batch074');assert not root.exists();slot=root/n/m;old=p['scratch_cwd'];scratch=str(slot/'scratch');fresh=copy.deepcopy(p)
  fields=['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']
  for k in fields:fresh[k]=move(p[k],old,scratch)
  readonly={str(Path(k).resolve()):v for k,v in p['execution_bindings'].items()};readonly.update({str(Path(k).resolve()):v for k,v in pr.get('additional_current_profile_input_bindings',{}).items()});artifacts=[];inversebind={}
  if n=='slu':
   compressed=cap/'build_manifest.json.gz';raw=gzip.decompress(compressed.read_bytes());oldmp=Path(p['build_command'][p['build_command'].index('--build-manifest')+1]);expected=p['build_command'][-1];assert hashlib.sha256(raw).hexdigest()==expected==p['execution_bindings'][str(oldmp)];assert not oldmp.exists()or oldmp.read_bytes()==raw
   orig=O/'build_manifests/original018.json';orig.parent.mkdir(exist_ok=True);orig.write_bytes(raw);newraw=raw.replace(old.encode(),scratch.encode());assert newraw.replace(scratch.encode(),old.encode())==raw;newmp=O/'build_manifests/fresh074.json';newmp.write_bytes(newraw);man=json.loads(newraw);assert man['scratch']==scratch and man['compiler_only']is True
   for key,h in man['input_bindings'].items():assert sha(key)==h
   for cmd in man['commands']:
    dest=cmd[2]if Path(cmd[0]).name=='ar'else cmd[cmd.index('-o')+1];assert Path(dest).is_relative_to(scratch)
   for key in ['final_outputs','must_be_absent_before_build']:assert all(Path(v).is_relative_to(scratch)for v in man[key])
   futuremp=str(slot/'build_manifest.json');fresh['build_command']=move(fresh['build_command'],str(oldmp),futuremp);fresh['build_command'][-1]=sha(newmp);assert fresh['execution_bindings'].pop(str(oldmp))==expected;fresh['execution_bindings'][futuremp]=sha(newmp);readonly.pop(str(oldmp.resolve()));readonly[str(orig)]=sha(orig);readonly[str(newmp)]=sha(newmp);readonly[str(compressed)]=sha(compressed);artifacts=[{'frozen_copy':info(newmp),'future_copy':futuremp,'future_sha256':sha(newmp)}];inversebind={str(oldmp):expected}
   save(O/'build_manifest_inverse_proof.json',{'retained_compressed':info(compressed),'original_full_bytes':info(orig),'original_captured_binding_SHA':expected,'fresh_full_bytes':info(newmp),'only_literal_scratch_prefix_replacement':True,'full_byte_inverse_equal':True,'commands':len(man['commands']),'input_bindings':len(man['input_bindings']),'no_compilation':True,'all_output_destinations_fresh_scratch':True})
  for q,h in readonly.items():assert sha(q)==h,(n,m,q)
  for q,h in p['execution_bindings'].items():
   fp=Path(q)
   if fp.suffix in ['.cu','.cuh','.cpp','.h','.c','.jl','.py']:
    dst=O/'source_snapshots'/h;dst.parent.mkdir(exist_ok=True)
    if not dst.exists():dst.write_bytes(fp.read_bytes())
    assert sha(dst)==h;sourceproof.append({'job':n+'__'+m,'original':q,'original_sha256':h,'frozen':info(dst),'source_edits':False,'full_source_byte_inverse_identity':True})
  rev=copy.deepcopy(fresh)
  if n=='slu':
   del rev['execution_bindings'][futuremp];rev['execution_bindings'].update(inversebind);rev['build_command']=move(rev['build_command'],futuremp,str(oldmp));rev['build_command'][-1]=expected
  for k in fields:rev[k]=move(rev[k],scratch,old)
  assert rev==p
  op=O/'original_plans'/f'{n}__{m}.json';op.parent.mkdir(exist_ok=True);op.write_bytes(oldraw);np=O/'plans'/f'{n}__{m}'/'plan.json';save(np,fresh)
  se=pr['selection_override'];assert sha(se['path'])==se['sha256'];selector=O/'selectors'/f'{n}__{m}.json';selector.parent.mkdir(exist_ok=True);selector.write_bytes(Path(se['path']).read_bytes());readonly[str(selector)]=sha(selector)
  if n=='dispatch':selectors.append(load(selector))
  for pp in [cap/'plan.json.gz',cap/'profile_provenance.json.gz',cap/'manifest.json',cap/'execution.log.gz']:readonly[str(pp)]=sha(pp)
  readonly[pr['profiler_executable']]=pr['profiler_executable_sha256'];timeout=3600 if n=='dispatch'else 900
  j={'job_id':n+'__'+m,'group':n,'benchmark':n,'model':m,'expected_contract':{k:ct[k]for k in ['configuration_id','raw_config_sha256','result_sha256']},'expected_latest_profile_id':pid,'original_plan':info(op),'prepared_plan':info(np),'plan_root':str(root),'future_plan_path':str(slot/'plan.json'),'scratch':scratch,'old_scratch':old,'timeout_seconds':timeout,'method':None,'readonly_bindings':readonly,'source_copies':[],'stage_artifacts':artifacts,'issues':[],'selection':info(selector),'full_inverse_original_plan_equal':True,'argv':['/usr/bin/python3',str(W/'tools/v100_roofline_collect.py'),'--source-root',str(W),'--plan-root',str(root),'--benchmark',n,'--skip-qualified-selected','--timeout',str(timeout),'--reserve-mib','2048','--selection-overrides',str(selector)]};jobs.append(j);proofs.append({'job_id':j['job_id'],'whole_original_plan_inverse_equal':True,'original_raw_bytes':info(op),'source_changes':False,'manifest_inverse_proof':'build_manifest_inverse_proof.json'if n=='slu'else None})
 # Each captured021 selector is already all4. Require exact bytes/objects instead of inventing union metadata.
 assert all(x==selectors[0]for x in selectors);assert set(selectors[0]['selection'])=={'cuda','cuda_jl','ka','jacc'}
 ap=O/'selectors/dispatch_all4_exact.json';ap.write_bytes(Path(jobs[0]['selection']['path']).read_bytes())
 for j in jobs:
  if j['group']=='dispatch':j['selection']=info(ap);j['readonly_bindings'][str(ap)]=sha(ap);j['argv'][-1]=str(ap)
 tools={str(W/p):sha(W/p)for p in ['tools/v100_roofline_collect.py','tools/v100_roofline_classify.py','tools/v100_profile_write_guard.py','tools/v100_profiler_method.py']};assert tools[str(W/'tools/v100_roofline_collect.py')]=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116'
 save(O/'job_inventory.json',{'jobs':jobs,'tool_bindings':tools,'inverse_proofs':proofs,'source_proofs':sourceproof,'canonical_SH_join_checked':True,'raw_roots_created':False,'GPUlaunches':0})
print('5 fullplan inverses and exact current source/config/result/selector/methodjoins pass; SLU manifest fullbyteinverse pass; no staging/compilation')
