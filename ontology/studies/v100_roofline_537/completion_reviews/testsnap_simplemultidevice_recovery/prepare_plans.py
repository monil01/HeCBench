"""CPU-only exact source-copy plan assembly. Never builds/launches/queries GPUs."""
from pathlib import Path
import json,gzip,hashlib,fcntl,copy,difflib,re,shutil,os
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).parent.resolve();R=Path('/tmp/hecbench-roofline-v100-20261020');UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
sha=lambda b:hashlib.sha256(b).hexdigest()
def H(p):return sha(Path(p).read_bytes())
def binding(p):return {'path':str(p),'sha256':H(p)}
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def replace_tree(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [replace_tree(v,a,b)for v in x]
 if isinstance(x,dict):return {k:replace_tree(v,a,b)for k,v in x.items()}
 return x
plans=[];allpatches=[];pins={};slots=[]
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for bench,models,batch in [('testSNAP',['cuda_jl','ka','jacc'],'batch067-testSNAP-observer-path'),('simpleMultiDevice',['cuda','cuda_jl','ka','jacc'],'batch068-simpleMultiDevice-GPU6UUID')]:
  root=R/batch
  if root.exists():raise RuntimeError('Fresh output root required: '+str(root))
  root.mkdir();ledgerpath=S/'benchmarks'/f'{bench}.json';ledger=json.loads(ledgerpath.read_bytes());contract=ledger['accepted_named_contract'];src=W/'src'/f'{bench}-julia'
  for model in models:
   manifest=W/ledger['models'][model]['profile_evidence_manifest'];parent=manifest.parent;old=load(parent/'plan.json.gz');provenance=load(parent/'profile_provenance.json.gz');original=copy.deepcopy(old)
   assert all(old[a]==contract[b]for a,b in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]);assert H(old['reference_result'])==old['reference_result_sha256'];assert H(Path(old['reference_result']).with_name('config.json'))==old['raw_config_sha256']
   for path,h in old['execution_bindings'].items():assert H(path)==h,(path,'captured source drift')
   slot=root/bench/model;slot.mkdir(parents=True);scratch=slot/'scratch';scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();newsrc=scratch/'profile_source';newsrc.mkdir();p=copy.deepcopy(old)
   for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:p[k]=replace_tree(old[k],old['scratch_cwd'],str(scratch))
   p['environment_overrides']['CUDA_VISIBLE_DEVICES']='6' # collector validates numeric plan, then pins UUID at target runtime
   patchlist=[]
   def patch(a,d,pairs=(),whole=None):
    a=a.resolve();raw=a.read_bytes();assert old['execution_bindings'].get(str(a))==sha(raw),(a,'not captured original')
    before=raw.decode();after=before
    if whole is not None:after=whole
    for x,y in pairs:
     assert after.count(x)==1,(a,x);after=after.replace(x,y,1)
    # Prove inverse of every literal substitution (or whole old guard replacement).
    inverse=after
    if whole is None:
     for x,y in reversed(pairs):assert inverse.count(y)==1,(a,y);inverse=inverse.replace(y,x,1)
    else:inverse=before
    assert inverse.encode()==raw
    d.write_text(after);d.chmod(0o444);entry={'source':str(a),'source_sha256':sha(raw),'copy':str(d),'copy_sha256':H(d),'text':after,'literal_substitutions':[{'before':x,'after':y}for x,y in pairs],'whole_guard_replacement':whole is not None,'original_text':before if whole is not None else None,'inverse_original_sha256':sha(inverse.encode()),'diff':''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(a),tofile=str(d)))};patchlist.append(entry);p['execution_bindings'][str(d)]=H(d);return d
   command=p['profile_target_command'];build=p['build_command']
   if bench=='testSNAP':
    common=newsrc/'study_common.jl';patch(src/'study_common.jl',common,[('const ROOT=normpath(@__DIR__)','const ROOT='+json.dumps(str(src))),('capture=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/testSNAP_primary"))','capture=joinpath(ENV["STUDY_OUTPUT_DIR"],"private_capture")')]);entry=Path(next(x for x in command if x.endswith('/main.jl')));copyentry=newsrc/'main.jl';patch(entry,copyentry,[('include(joinpath(@__DIR__,"..","study_common.jl"))','include('+json.dumps(str(common))+')'),('include(joinpath(@__DIR__,"..","study_device.jl"))','include('+json.dumps(str(src/'study_device.jl'))+')')]);command[command.index(str(entry))]=str(copyentry)
   elif model=='cuda':
    a=src/'study_native.cu';text=a.read_text();oldguard=next(l for l in text.splitlines()if l.startswith('void guard('));newguard='void guard(bool initial=false){(void)initial;const char*v=getenv("CUDA_VISIBLE_DEVICES");if(!v||strcmp(v,"'+UUID+'"))exit(3);int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count!=1)exit(3);cudaDeviceProp prop;if(cudaGetDeviceProperties(&prop,0)!=cudaSuccess)exit(3);const unsigned char expected[16]={0xc1,0x22,0xec,0xbd,0xd9,0xf3,0x18,0xa5,0x42,0xd4,0x15,0x35,0xaf,0x37,0xe5,0x6d};if(memcmp(prop.uuid.bytes,expected,16))exit(3);}'
    d=newsrc/'study_native.cu';patch(a,d,[(oldguard,newguard),('#include "../simpleMultiDevice-cuda/main.cu"','#include '+json.dumps(str(W/'src/simpleMultiDevice-cuda/main.cu')))]);assert build.count(str(a))==1;build[build.index(str(a))]=str(d);assert command[-1]=='1'
   else:
    guard=newsrc/'study_guard.jl';guardtext='function study_guard(;initial=false)\n get(ENV,"CUDA_VISIBLE_DEVICES","") == "'+UUID+'" || error("Only exact authorized GPU6 UUID visibility")\n nothing\nend\nstudy_guard(initial=true)\n';patch(src/'study_guard.jl',guard,whole=guardtext)
    common=newsrc/'study_common.jl';needle='g=parse(Int,ARGS[1]);devices=collect(CUDA.devices());length(devices)==g||error("Actualvisibledevicecount differs")';added=needle+'\n g==1 || error("GPU6-only profiling requires argv 1")\n replace(lowercase(string(CUDA.uuid(devices[1]))),"gpu-"=>"")=="c122ecbd-d9f3-18a5-42d4-1535af37e56d" || error("GPU6 runtime UUID mismatch")';patch(src/'study_common.jl',common,[(needle,added),('p=joinpath(@__DIR__,"oracle_g$(g)_u64.bin")','p=joinpath('+json.dumps(str(src))+',"oracle_g$(g)_u64.bin")')]);entry=Path(next(x for x in command if x.endswith('/main.jl')));d=newsrc/'main.jl';patch(entry,d,[('include(joinpath(@__DIR__,"..","study_guard.jl"))','include('+json.dumps(str(guard))+')'),('include(joinpath(@__DIR__,"..","study_common.jl"))','include('+json.dumps(str(common))+')')]);command[command.index(str(entry))]=str(d);assert command[-1]=='1'
   # Explicit absolute user-space Julia executable, same recorded Julia launcher.
   if command[0]=='julia':command[0]='/home/imo/.juliaup/bin/julia'
   assert not p['blocking_gates'];assert p['environment_overrides']['CUDA_VISIBLE_DEVICES']=='6'
   for asset in p['runtime_assets']:
    for f in asset['source_files']:assert H(f['path'])==f['sha256']
    a=Path(asset['source']);d=Path(asset['destination']);assert d.is_relative_to(scratch);d.parent.mkdir(parents=True,exist_ok=True)
    if a.is_dir():shutil.copytree(a,d)
    else:shutil.copy2(a,d)
    for f in ([d]if d.is_file()else list(d.rglob('*'))):
     if f.is_file():f.chmod(0o444)
   # Bind exact historical input plan/provenance and metadata assembly source.
   for f in [parent/'plan.json.gz',parent/'profile_provenance.json.gz',manifest,Path(__file__).resolve()]:p['execution_bindings'][str(f)]=H(f)
   review={'kind':'testSNAP_Julia_observer_output_path_only'if bench=='testSNAP'else 'simpleMultiDevice_exact_GPU6_UUID_guard_only','kernel_math_checks_iteration_timer_changes':False,'new_guard_outside_timing':bench=='simpleMultiDevice','native_runtime_guard': 'exact environment UUID, CUDA count1 and raw 16-byte UUID before/all five trials/after checks'if model=='cuda'and bench=='simpleMultiDevice'else None,'Julia_runtime_guard':'exact environment before CUDA load, count1/argv1 and UUID after device enumeration before allocations/warmup'if model!='cuda'and bench=='simpleMultiDevice'else None,'GPU5_branch_or_query_in_executed_overlay':False,'sample_phase':'unknown','whole_ROI_claim':False,'replay_adaptation_is_not_observed_fix':True,'all_original_5_checks_required':True,'primary_timing_unchanged':True,'root_review_launch_required':True}
   p['capture_overlay_provenance']={'patches':patchlist,'historical_plan':binding(parent/'plan.json.gz'),'historical_provenance':binding(parent/'profile_provenance.json.gz'),'reference_result_sha256':p['reference_result_sha256'],'reference_config_sha256':p['raw_config_sha256'],'review':review};p['staged']=True;p['commands_executed']={'build':False,'target':False,'GPU':False};p['review']=dict(p.get('review') or {},narrow_recovery=review);p['notes']+=['Private proposed source-equivalent isolation/UUID guard recovery; root review required. No build/target execution.']
   save(slot/'plan.json',p);save(slot/'original_plan.json',original);save(slot/'plan_diff.json',{k:{'before':original.get(k),'after':v}for k,v in p.items()if original.get(k)!=v});pins[str(ledgerpath)]=H(ledgerpath)
   for path,h in p['execution_bindings'].items():assert H(path)==h
   rows={'benchmark':bench,'model':model,'plan':binding(slot/'plan.json'),'historical_plan':binding(parent/'plan.json.gz'),'reference_result_sha256':p['reference_result_sha256'],'configuration_id':p['configuration_id'],'historical_status':provenance['status'],'source_patch_count':len(patchlist),'review':review};plans.append(rows);allpatches+=patchlist
 fcntl.flock(lock,fcntl.LOCK_UN)
save(O/'plans.json',{'plans':plans,'source_patches':allpatches,'selected_ledger_bindings':pins,'GPU_or_build_execution':False});print(json.dumps({'plans':len(plans),'patches':len(allpatches),'roots':[str(R/x)for x in ['batch067-testSNAP-observer-path','batch068-simpleMultiDevice-GPU6UUID']]}))
