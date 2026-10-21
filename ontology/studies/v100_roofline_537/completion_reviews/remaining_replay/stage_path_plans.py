#!/usr/bin/env python3
"""CPU-only path copies for exact remaining reviewed single-GPU contracts.
Never builds, runs targets, touches GPUs or authorizes replay. Fresh /tmp only.
"""
import argparse,pathlib,hashlib,json,copy,re,shutil,difflib
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');BASE=P('/tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512')
FAMILIES={'p4':['cuda','cuda_jl','ka','jacc'],'particlefilter':['cuda'],'pointwise':['cuda'],'recursiveGaussian':['cuda'],'rowwiseMoments':['cuda'],'spsm':['cuda','cuda_jl','ka','jacc'],'stsg':['cuda'],'remap':['cuda','cuda_jl','ka','jacc'],'sc':['cuda','cuda_jl','ka','jacc']}
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
   if n=='p4':
    if m=='cuda':
     f=root/'study_cuda.cu';dest=patch(f,[('std::filesystem::path(__FILE__).parent_path()','std::filesystem::path('+json.dumps(str(root))+')'),('"/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/p4"',json.dumps(str(priv/'p4')))])
    else:
     common=patch(root/'study_common.jl',[('"/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/p4"',json.dumps(str(priv/'p4')))],julia=True)
   elif n in ['particlefilter','pointwise','recursiveGaussian','rowwiseMoments']:
    prefix={'particlefilter':'particlefilter_','pointwise':'pointwise_','recursiveGaussian':'recursiveGaussian_','rowwiseMoments':'rowwise_'}[n];f=root/'study_native.cu';dest=patch(f,[(json.dumps('/tmp/'+prefix),json.dumps(str(scratch/'profile_outputs'/prefix)))])
   elif n=='spsm':
    if m=='cuda':
     f=root/'study_native.cu';dest=patch(f,[('"/home/imo/HeCBench_julia/.porting-state/portable_julia/spsm_reference_review/native_warm_m"',json.dumps(str(priv/'native_warm_m')))])
    else:common=patch(root/'study_common.jl',[('"/home/imo/HeCBench_julia/.porting-state/portable_julia/spsm_reference_review"',json.dumps(str(priv)))],julia=True)
   elif n=='stsg':
    f=root/'study_cuda.cu';dest=patch(f,[('"/home/imo/HeCBench_julia/.porting-state/porting_logs/stsg-julia-shared/native_failed_warmup_candidate02.bin"',json.dumps(str(priv/'native_failed_warmup_candidate02.bin')))])
   if common:
    entry=P(next(x for x in plan['profile_target_command'] if x.endswith('.jl')));text=entry.read_text();before='include(joinpath(@__DIR__,"..","study_common.jl"))'
    if before not in text:before='include("../study_common.jl")'
    dest=patch(entry,[(before,'include('+json.dumps(str(common))+')')],label='main_entry.jl',julia=True);plan['profile_target_command']=[str(dest)if x==str(entry)else x for x in plan['profile_target_command']]
   elif patches:
    if plan['build_command'].count(str(f))!=1:raise ValueError('Unique adapter build arg needed')
    plan['build_command']=[str(dest)if x==str(f)else x for x in plan['build_command']]
   cleared=[]
   if patches:
    gate='source_bound_absolute_path_literals_require_write_isolation_review';cleared=[x for x in plan['blocking_gates']if x==gate];plan['blocking_gates']=[x for x in plan['blocking_gates']if x!=gate]
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
