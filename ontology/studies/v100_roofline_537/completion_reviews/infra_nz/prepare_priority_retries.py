#!/usr/bin/env python3
"""CPU-only fresh path/launcher overlays for reviewed pointerchase and Sparkler slots.
No compiler, Julia, MPI, profiler, GPU query or target is executed. Requires the
immutable review.json beside this helper and a fresh strictly-/tmp output root.
"""
import argparse,copy,hashlib,json,pathlib,shutil,difflib
P=pathlib.Path
UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def dump(p,a):P(p).write_text(json.dumps(a,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--benchmark',choices=['pointerchase','sparkler'],action='append');ap.add_argument('--stage',action='store_true');a=ap.parse_args()
 review=P(__file__).with_name('review.json');r=json.loads(review.read_text());out=a.output_root.absolute()
 if out.exists() or out.is_symlink() or not out.resolve().is_relative_to(P('/tmp')) or out==P('/tmp'):ap.error('Fresh /tmp output root required')
 chosen=[x for x in r['slots'] if x['benchmark'] in (a.benchmark or ['pointerchase','sparkler']) and x.get('latest_raw_provenance')]
 # Validate all selected plans and canonical input bindings before creating output.
 for x in chosen:
  bp=P(x['base_plan']['path'])
  if sha(bp)!=x['base_plan']['sha256']:raise ValueError('Frozen base plan drift')
  b=json.loads(bp.read_text())
  frozen=b.get('host_overlay_provenance',b.get('overlay_provenance',{}))
  stored={z['copy']:z for z in frozen.get('patches',frozen.get('profile_source_patches',[]))}
  for p,h in b['execution_bindings'].items():
   if p in stored:
    z=stored[p]
    if h!=z['copy_sha256'] or hashlib.sha256(z['text'].encode()).hexdigest()!=h:raise ValueError('Frozen overlay binding mismatch: '+p)
   elif not P(p).is_file() or sha(p)!=h:raise ValueError('Canonical input drift: '+p)
  for q in x['review_source_bindings']:
   if sha(q['path'])!=q['sha256']:raise ValueError('Reviewed adapter drift')
 out.mkdir(parents=True);summary=[]
 for x in chosen:
  n,m=x['benchmark'],x['model'];bp=P(x['base_plan']['path']);base=json.loads(bp.read_text());slot=out/n/m;scratch=slot/'scratch';src=scratch/'profile_source';slot.mkdir(parents=True)
  oldslot=str(bp.parent);old=base['scratch_cwd']
  def rebase(v):
   if isinstance(v,str):return v.replace(oldslot,str(slot))
   if isinstance(v,list):return [rebase(z) for z in v]
   if isinstance(v,dict):return {k:rebase(z) for k,z in v.items()}
   return v
  plan=copy.deepcopy(base)
  for k in ['scratch_cwd','build_command','profile_target_command','environment_overrides','runtime_assets']:plan[k]=rebase(base[k])
  if plan['blocking_gates']:raise ValueError('Do not clear unreviewed gates')
  if plan['runtime_asset_bytes']:raise ValueError('Unexpected runtime asset copies; separate budget review needed')
  patches=[];additional={str(review):sha(review),str(P(__file__).resolve()):sha(__file__),str(bp):sha(bp)}
  def add(original,dest,text,before=None):
   original=P(original);originaltext=original.read_text() if before is None else before
   additional[str(original)]=sha(original)
   patches.append({'source':str(original),'source_sha256':sha(original),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'diff':''.join(difflib.unified_diff(originaltext.splitlines(True),text.splitlines(True),fromfile=str(original),tofile=str(dest)))})
  if n=='pointerchase':
   canonical=P(x['source_directory']);common=canonical/'study_common.jl';text=common.read_text();before='root=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","pointerchase-julia-"*model,"diagnostics")';after='root=joinpath(ENV["STUDY_OUTPUT_DIR"],"pointerchase_diagnostics",model)'
   if text.count(before)!=1:raise ValueError('Pointer diagnostic root occurrence mismatch')
   add(common,src/common.name,text.replace(before,after,1));cmd=plan['profile_target_command'];entry=P(cmd[-1]);et=entry.read_text();inc='include("../study_common.jl")'
   if et.count(inc)!=1:raise ValueError('Pointer include occurrence mismatch')
   add(entry,src/'main.jl',et.replace(inc,'include('+json.dumps(str(src/common.name))+')',1));cmd[-1]=str(src/'main.jl')
  else:
   # Rehydrate already-reviewed frozen isolation overlays, retain all4MPI ranks.
   provenance=base.get('host_overlay_provenance',base.get('overlay_provenance',{}))
   for patch in provenance.get('patches',provenance.get('profile_source_patches',[])):
    text=patch['text']
    if hashlib.sha256(text.encode()).hexdigest()!=patch['copy_sha256']:raise ValueError('Frozen overlay text mismatch')
    original=P(patch.get('original',patch.get('source')))
    if sha(original)!=patch.get('original_sha256',patch.get('source_sha256')):raise ValueError('Frozen overlay original drift')
    add(original,P(rebase(patch['copy'])),rebase(text))
   launcher=P(x['source_directory'])/'study_launch.py';text=launcher.read_text();before="if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise SystemExit('Require recorded GPU6-only visibility')";after="if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('6', '"+UUID+"'):raise SystemExit('Require recorded physical GPU6-only visibility')"
   if text.count(before)!=1:raise ValueError('Sparkler launcher occurrence mismatch')
   dest=src/launcher.name;add(launcher,dest,text.replace(before,after,1));cmd=plan['profile_target_command']
   if cmd.count(str(launcher))!=1:raise ValueError('Sparkler launcher argument mismatch')
   cmd[cmd.index(str(launcher))]=str(dest)
  additional.update({q['path']:q['sha256'] for q in x['review_source_bindings']})
  bindings={}
  for path,h in base['execution_bindings'].items():
   target=rebase(path);match=next((z for z in patches if z['copy']==target),None)
   bindings[target]=match['copy_sha256'] if match else h
  for patch in patches:bindings[patch['copy']]=patch['copy_sha256']
  plan['execution_bindings']=bindings
  if plan.get('review') is None:plan['review']={}
  plan['review']['additional_current_readonly_bindings']=additional
  plan['staged']=bool(a.stage);plan['completion_retry_proposal']={'base_plan':str(bp),'base_plan_sha256':sha(bp),'changes':'host paths and exact GPU6 UUID visibility spelling only','compiler_runs':0,'GPU_runs':0,'replay_safety_approval':False,'patches':patches,'additional_current_readonly_bindings':additional}
  plan['commands_executed']=[]
  if a.stage:
   (scratch/'profile_outputs').mkdir(parents=True);(scratch/'ncu_tmp').mkdir();src.mkdir(exist_ok=True)
   for patch in patches:
    dest=P(patch['copy'])
    if not dest.resolve().is_relative_to(scratch.resolve()):raise ValueError('Copy escape')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(patch['text'])
   for p in provenance.get('precreated_directories',[]) if n=='sparkler' else []:
    q=P(rebase(p))
    if q.resolve().is_relative_to(scratch.resolve()):q.mkdir(parents=True,exist_ok=True)
  dump(slot/'plan.json',plan);dump(slot/'patches.json',patches);summary.append({'benchmark':n,'model':m,'plan':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'staged':bool(a.stage)})
 dump(out/'summary.json',summary)
if __name__=='__main__':main()
