#!/usr/bin/env python3
"""CPU-only path copies for exact remaining reviewed single-GPU contracts.
Never builds, runs targets, touches GPUs or authorizes replay. Fresh /tmp only.
"""
import argparse,pathlib,hashlib,json,copy,re,shutil,difflib,ast,sys
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');BASE=P('/tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512')
FAMILIES={'tsne':['cuda']}
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def dump(p,a):P(p).write_text(json.dumps(a,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-root',type=P,required=True);a=ap.parse_args();out=a.output_root.resolve()
 if out.exists()or not out.is_relative_to('/tmp')or out==P('/tmp'):ap.error('Fresh /tmp output required')
 out.mkdir();summary=[]
 for n,models in FAMILIES.items():
  for m in models:
   bp=BASE/n/m/'plan.json';b=json.loads(bp.read_text());slot=out/n/m;scratch=slot/'scratch';priv=scratch/'profile_outputs'/'private_capture';src=scratch/'profile_source';slot.mkdir(parents=True)
   for p,h in b['execution_bindings'].items():
    if sha(p)!=h:raise ValueError('Canonical drift '+p)
   rp=P(b['reference_result']);cp=rp.with_name('config.json')
   if sha(rp)!=b['reference_result_sha256']or sha(cp)!=b['raw_config_sha256']:raise ValueError('Reference/config drift')
   old=b['scratch_cwd'];plan=copy.deepcopy(b)
   def move(v):
    if isinstance(v,str):return v.replace(old,str(scratch))
    if isinstance(v,list):return [move(x)for x in v]
    if isinstance(v,dict):return {k:move(x)for k,x in v.items()}
    return v
   for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:plan[k]=move(b[k])
   root=W/'src'/(n+'-julia');patches=[];bindings=dict(b['execution_bindings']);bindings[str(bp)]=sha(bp);bindings[str(P(__file__).resolve())]=sha(__file__)
   def patch(p,pairs,label=None,julia=False):
    original=p.read_text();text=original;bindings[str(p)]=sha(p)
    for before,after in pairs:
     if text.count(before)!=1:raise ValueError('Exact occurrence mismatch '+str(p)+' '+before)
     text=text.replace(before,after,1)
    if julia:
     text=text.replace('@__DIR__',json.dumps(str(p.parent)))
     text=re.sub(r'include\("([^"\n]+)"\)',lambda z:'include('+json.dumps(str((p.parent/z[1]).resolve()))+')',text)
    else:
     def include(z):
      target=(p.parent/z[2]).resolve();bindings[str(target)]=sha(target);return z[1]+json.dumps(str(target))
     text=re.sub(r'(^\s*#\s*include\s*)"([^"]+)"',include,text,flags=re.M)
    dest=src/(label or p.name);patches.append({'source':str(p),'source_sha256':sha(p),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'diff':''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(p),tofile=str(dest)))});return dest
   common=None
   if n=='tsne':
    f=root/'run_metric.py';wrapper=patch(f,[('from power2_codec import convert,digest','sys.path.insert(0,'+json.dumps(str(root))+');from power2_codec import convert,digest'),('P=pathlib.Path(__file__).resolve().parent','P=pathlib.Path('+json.dumps(str(root))+')'),("'/tmp/hecbench-portable-infrastructure-private/tsne/metric'",repr(str(priv/'tsne'/'metric'))),("'/tmp/hecbench-portable-infrastructure-private/tsne'",repr(str(priv/'tsne')))])
    plan['profile_target_command']=[str(wrapper)if x==str(f)else x for x in plan['profile_target_command']]
    f=root/'guarded_I1/study_native.cu';dest=patch(f,[('"/tmp/hecbench-portable-infrastructure-private/tsne/native/pid"',json.dumps(str(priv/'tsne'/'native'/'pid'))),('"/tmp/tsne-native-capture"',json.dumps(str(scratch/'profile_outputs')))])
    log=P('/tmp/tsne-guarded-I1-native-build01.log');recipe=root/'guarded_I1/build_native.py';oldcmd=ast.literal_eval(log.read_text().splitlines()[0]);expected=['/usr/bin/nvcc','-O3','--use_fast_math','-std=c++17','-arch=sm_70','-allow-unsupported-compiler','-include',str(root/'guarded_I1/guard_allocator.h'),'-I'+str(W/'src/tsne-cuda'),str(f),*[str(q)for q in sorted((W/'src/tsne-cuda').glob('*.cu'))if q.name not in['main.cu','fit_tsne.cu']],'-o','/tmp/tsne-guarded-I1-native']
    if oldcmd!=expected:raise ValueError('Exact frozen compiler recipe/source TU set mismatch')
    if bindings.get(str(recipe))!=sha(recipe):raise ValueError('Selected exact recipe source binding mismatch')
    for q in [recipe,log,P(oldcmd[0]),root/'guarded_I1/guard_allocator.h']:
     bindings[str(q)]=sha(q)
    for directory in [root,W/'src/tsne-cuda']:
     for q in directory.rglob('*'):
      if q.is_file() and q.suffix in ['.h','.hpp','.cu','.cuh','.inc','.cpp','.py','.jl']:bindings[str(q)]=sha(q)
    compiler_manifest={'original_recipe':str(recipe),'original_recipe_sha256':sha(recipe),'frozen_build_log':str(log),'frozen_build_log_sha256':sha(log),'original_command':oldcmd,'compiler_current_prospective_sha256':sha(oldcmd[0]),'historical_compiler_hash_observed':False,'qualification':'Original exactselectedrecipe source andfrozenargv proven. Currentcompiler/toolkit sourcebindings prospective only, nothistoricalcapture. No compilationexecuted.'}
    binary=scratch/'study_native';newcmd=[str(dest)if x==str(f)else str(binary)if x=='/tmp/tsne-guarded-I1-native'else x for x in oldcmd]
    guard=W/'tools/v100_profile_write_guard.py';bindings[str(guard)]=sha(guard);plan['build_command']=[sys.executable,str(guard),'--allow-dir',str(scratch),'--',*newcmd]
    plan['profile_target_command']=[str(binary)if x=='/tmp/tsne-guarded-I1-native'else x for x in plan['profile_target_command']]
    compiler_manifest['copied_adapter_command']=newcmd;compiler_manifest['guarded_build_command']=plan['build_command'];plan['exact_native_recipe_proof']=compiler_manifest
   if common:
    entry=P(next(x for x in plan['profile_target_command'] if x.endswith('.jl')));text=entry.read_text();before='include(joinpath(@__DIR__,"..","study_common.jl"))'
    if before not in text:before='include("../study_common.jl")'
    dest=patch(entry,[(before,'include('+json.dumps(str(common))+')')],label='main_entry.jl',julia=True);plan['profile_target_command']=[str(dest)if x==str(entry)else x for x in plan['profile_target_command']]
   oracle='/tmp/tsne_oracle_mnist_i1';boundoracle=[p for p in bindings if p.startswith(oracle+'/')]
   if len(boundoracle)!=32:raise ValueError('Exact oracle32files receipt required')
   cleared=[]
   if patches:
    gate='source_bound_absolute_path_literals_require_write_isolation_review';cleared=[x for x in plan['blocking_gates']if x==gate];plan['blocking_gates']=[x for x in plan['blocking_gates']if x!=gate]
    for role in ['absolute_directory_argument_requires_role:/tmp/tsne_oracle_mnist_i1','unbound_absolute_file_argument_requires_role:/home/imo/.juliaup/bin/julia']:
     if role in plan['blocking_gates']:cleared.append(role);plan['blocking_gates'].remove(role)
   if plan['blocking_gates']:raise ValueError('Unreviewed gates '+str(plan['blocking_gates']))
   # Native and Julia patches retain failed-oracle diagnostics and exact failure behavior.
   # Explicit readonly fixture literals and cuSPARSE exact-library path remain unchanged.
   if shutil.disk_usage(out).free<plan['runtime_asset_bytes']+2*1024**3:raise ValueError('Reserve insufficient')
   scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();priv.mkdir()
   for asset in plan['runtime_assets']:
    for q in asset['source_files']:
     if sha(q['path'])!=q['sha256']:raise ValueError('Asset drift')
    source=P(asset['source']);destination=P(asset['destination']);destination.relative_to(scratch);destination.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir():shutil.copytree(source,destination)
    else:shutil.copy2(source,destination)
   for q in patches:
    d=P(q['copy']);d.parent.mkdir(parents=True,exist_ok=True);d.write_text(q['text']);bindings[str(d)]=sha(d)
   proof={'base_plan':str(bp),'base_plan_sha256':sha(bp),'source_result_sha256':b['reference_result_sha256'],'source_config_sha256':b['raw_config_sha256'],'patches':patches,'source_bindings':bindings,'cleared_gates':cleared,'kernel_math_changes':False,'preserves_all5_trials_and_oracle_failures':True,'GPU_runs':0,'build_runs':0,'replay_approval':False,'whole_ROI_claim':False}
   dump(slot/'source_proof.json',proof);bindings[str(slot/'source_proof.json')]=sha(slot/'source_proof.json');plan['execution_bindings']=bindings;plan['staged']=True;plan['staging_status']='input_plan_ready_ROI_and_kernel_mapping_still_required';plan['remaining_completion_proof']=proof;plan['commands_executed']=[];dump(slot/'plan.json',plan);summary.append({'benchmark':n,'model':m,'plan':str(slot/'plan.json'),'sha256':sha(slot/'plan.json')})
 dump(out/'summary.json',summary);print(json.dumps({'staged':len(summary),'GPU_runs':0,'build_runs':0}))
if __name__=='__main__':main()
