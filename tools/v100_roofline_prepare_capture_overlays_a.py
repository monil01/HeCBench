#!/usr/bin/env python3
"""Prepare only clock/linearprobing/pointerchase/rsc/rsmt capture-path overlays.
No build/target/GPU execution; fresh /tmp plans, copies, optional readonly inputs.
The imported utility module is immutable/hash-pinned and has no import side effects.
"""
import argparse,copy,difflib,hashlib,importlib.util,json,os,pathlib,re,shutil,sys
sys.dont_write_bytecode=True
P=pathlib.Path
SLOTS=[(b,m)for b in ('clock','linearprobing','pointerchase','rsc','rsmt')for m in ('cuda_jl','ka','jacc')]+[('linearprobing','cuda'),('rsmt','cuda')]
UTILITY_SHA='8f94014c8c6264cce04e48e0dee6f4f2eec7673a7e1c67c1502c0d6f068309ce'
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);a.add_argument('--base-plan-root',type=P,required=True);a.add_argument('--reviews',type=P,required=True);a.add_argument('--reviews-sha256',required=True);a.add_argument('--output-root',type=P,required=True);a.add_argument('--benchmark',action='append',choices=sorted({b for b,m in SLOTS}));a.add_argument('--stage',action='store_true');a.add_argument('--asset-cap-mib',type=int,default=512);a.add_argument('--reserve-mib',type=int,default=2048);z=a.parse_args();R=z.source_root.resolve();SELF=P(__file__).resolve();U=SELF.with_name('v100_roofline_prepare_host_overlays.py');G=SELF.with_name('v100_profile_write_guard.py')
 if hashlib.sha256(U.read_bytes()).hexdigest()!=UTILITY_SHA:raise ValueError('Frozen utility changed; review new helper version before use')
 spec=importlib.util.spec_from_file_location('readonly_host_overlay_utilities',U);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u);sha,check,dump,under=u.sha,u.check,u.dump,u.under
 O=z.output_root.absolute()
 if not under(O,'/tmp')or O.resolve()==P('/tmp')or O.exists()or O.is_symlink():a.error('Fresh owned output strictly under /tmp required')
 if min(z.asset_cap_mib,z.reserve_mib)<0:a.error('Nonnegative budgets required')
 if sha(z.reviews)!=z.reviews_sha256:raise ValueError('Review table hash mismatch')
 reviews={(r['benchmark'],r['model']):r for r in json.loads(z.reviews.read_text())['models']};O.mkdir(parents=True);rows=[]
 for b,m in SLOTS:
  if z.benchmark and b not in z.benchmark:continue
  review=reviews[b,m];f=z.base_plan_root/b/m/'plan.json';base=json.loads(f.read_text());p=copy.deepcopy(base);source=P(review['source']).resolve()
  if sha(f)!=review['source_plan_sha256']or base['reference_result_sha256']!=review['reference_result_sha256']or base['raw_config_sha256']!=review['reference_config_sha256']or sha(source)!=review['source_sha256']:raise ValueError('Exact review/plan/source join changed '+b+'/'+m)
  check(base['execution_bindings']);bound={str(P(k).resolve()):h for k,h in base['execution_bindings'].items()}
  if bound.get(str(source))!=sha(source):raise ValueError('Capture source not bound by selected accepted model')
  rp=P(base['reference_result']);cp=rp.with_name('config.json')
  if sha(rp)!=base['reference_result_sha256']or sha(cp)!=base['raw_config_sha256']:raise ValueError('Accepted evidence drift')
  slot=O/b/m;S=slot/'scratch';slot.mkdir(parents=True);old=base['scratch_cwd'];move=lambda x:x.replace(old,str(S))if isinstance(x,str)else x
  for k in ('build_command','profile_target_command'):p[k]=[move(x)for x in p[k]]
  p['scratch_cwd']=str(S);p['environment_overrides']={k:move(v)for k,v in p['environment_overrides'].items()}
  for asset in p['runtime_assets']:asset['destination']=move(asset['destination'])
  bindings={**base['execution_bindings'],str(SELF):sha(SELF),str(U):UTILITY_SHA,str(G):sha(G),str(z.reviews.resolve()):z.reviews_sha256,str(f.resolve()):sha(f),str(cp):sha(cp)};context={};patches=[]
  for directory in (R/'src'/(b+'-julia'),R/'src'/(b+'-cuda')):
   for q in directory.rglob('*'):
    if q.is_file()and q.suffix in{'.jl','.cu','.cpp','.cc','.c','.h','.cuh','.hpp','.py','.inc','.toml'}:context[str(q.resolve())]=sha(q)
  def patched(src,dest,changes):
   original=src.read_text();text=original
   if bound.get(str(src.resolve()))!=sha(src):raise ValueError('Patch input not in selected source bindings '+str(src))
   for x,y in changes:
    if text.count(x)!=1:raise ValueError('Expected one exact path fragment '+str(src))
    text=text.replace(x,y)
   patches.append({'original':str(src.resolve()),'original_sha256':sha(src),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'changes':[{'before':x,'after':y}for x,y in changes],'unified_diff':list(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(src),tofile=str(dest)))})
  dest=S/'profile_source'/source.name;oldassignment=review['exact_output_root_assignment'];variable=oldassignment.split('=',1)[0].strip();raw=S/'profile_outputs'/('private_'+b)
  if m=='cuda':
   newassignment=variable+'=std::filesystem::path('+json.dumps(str(raw))+')';changes=[(oldassignment,newassignment)]
   for line in source.read_text().splitlines():
    match=re.fullmatch(r'\s*#include\s+"([^"]+)"\s*',line)
    if match:changes.append((line,'#include '+json.dumps(str((source.parent/match[1]).resolve()))))
   patched(source,dest,changes)
   if p['build_command'].count(str(source))!=1:raise ValueError('Unique native source build argument required')
   p['build_command']=[str(dest)if x==str(source)else x for x in p['build_command']]
  else:
   changes=[(oldassignment,variable+'=joinpath(ENV["STUDY_OUTPUT_DIR"],'+json.dumps('private_'+b)+')')];text=source.read_text().replace(changes[0][0],changes[0][1])
   for line in text.splitlines():
    if '@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(source.parent)))))
    match=re.fullmatch(r'include\("([^"]+)"\)',line.strip())
    if match:changes.append((line,'include('+json.dumps(str((source.parent/match[1]).resolve()))+')'))
   patched(source,dest,changes)
   entries=[P(x)for x in p['profile_target_command']if x.endswith('.jl')]
   if len(entries)!=1:raise ValueError('Unique Julia entry required')
   entry=entries[0];lines=entry.read_text().splitlines();common=[l for l in lines if l.strip().startswith('include(')and'study_common.jl'in l]
   if len(common)!=1:raise ValueError('Unique shared capture module include required')
   changes=[(common[0],'include('+json.dumps(str(dest))+')')]
   for line in lines:
    if line==common[0]:continue
    if '@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(entry.parent)))))
    match=re.fullmatch(r'include\("([^"]+)"\)',line.strip())
    if match:changes.append((line,'include('+json.dumps(str((entry.parent/match[1]).resolve()))+')'))
   newentry=S/'profile_source/main.jl';patched(entry,newentry,changes);p['profile_target_command']=[str(newentry)if x==str(entry)else x for x in p['profile_target_command']]
  # These were hidden output roots, not a basis to clear any unrelated old gate.
  if p['runtime_asset_bytes']>z.asset_cap_mib*1024**2:p['blocking_gates'].append('capture_overlay_asset_budget_exceeded')
  if p['build_command']:
   compiler=list(p['build_command']);p['build_command']=[sys.executable,str(G),'--allow-dir',str(S),'--',*compiler];dump(slot/'compiler_manifest.json',{'compiler_command':compiler,'guarded_build_command':p['build_command'],'guard_sha256':sha(G),'compiler_runs':0,'GPU_runs':0});bindings[str(slot/'compiler_manifest.json')]=sha(slot/'compiler_manifest.json')
  p['execution_bindings']=bindings;p['staged']=False;p['staging_status']='blocked'if p['blocking_gates']else'input_plan_ready_ROI_and_kernel_mapping_still_required';p['profiler_requirements']['filesystem_write_guard_required']=True
  p['review']={'path':str(z.reviews.resolve()),'sha256':z.reviews_sha256,'entry':{'reason':'Only active capture root and canonical include/fixture path lookup changed; kernels/math/RNG/geometry/iterations and all other env unchanged.','review':review,'additional_current_readonly_input_bindings':context,'cleared_gates':[],'kernel_math_changes':False,'profile_only':True}}
  p['capture_overlay_provenance']={'planner_sha256':sha(SELF),'utility_sha256':UTILITY_SHA,'guard_sha256':sha(G),'base_plan_sha256':sha(f),'patches':patches,'source_before':{**base['execution_bindings'],**context},'source_after':None,'compiler_runs':0,'GPU_runs':0}
  if z.stage and not p['blocking_gates']:
   if shutil.disk_usage(O).free<p['runtime_asset_bytes']+z.reserve_mib*1024**2:raise ValueError('Free-space reserve prevents staging')
   S.mkdir();(S/'profile_outputs').mkdir();(S/'ncu_tmp').mkdir()
   for asset in p['runtime_assets']:
    src=P(asset['source']);target=P(asset['destination'])
    if not under(target,S):raise ValueError('Asset destination escape')
    for q in asset['source_files']:
     if sha(q['path'])!=q['sha256']:raise ValueError('Asset input changed')
    target.parent.mkdir(parents=True,exist_ok=True)
    if src.is_dir():shutil.copytree(src,target)
    else:shutil.copy2(src,target)
    for q in [target]if target.is_file()else[x for x in target.rglob('*')if x.is_file()]:q.chmod(0o555 if os.access(q,os.X_OK)else 0o444)
   for q in patches:
    target=P(q['copy']);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(q['text']);target.chmod(0o444)
    if sha(target)!=q['copy_sha256']:raise ValueError('Source copy differs')
    bindings[str(target)]=q['copy_sha256']
   check(bindings);check(context);p['capture_overlay_provenance']['source_after']=dict(p['capture_overlay_provenance']['source_before']);p['capture_overlay_provenance']['source_bytes_unchanged_after_staging']=True;p['staged']=True
  dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in p.items()if base.get(k)!=v});dump(slot/'plan.json',p);rows.append({'benchmark':b,'model':m,'plan':str(slot/'plan.json'),'staged':p['staged'],'blocking_gates':p['blocking_gates'],'patch_count':len(patches)})
 dump(O/'summary.json',{'models':rows,'ready':sum(not r['blocking_gates']for r in rows),'staged':sum(r['staged']for r in rows),'planner_sha256':sha(SELF),'compiler_runs':0,'GPU_runs':0,'source_edits':False,'only_new_helper_written_in_source':True,'ROI_and_replay':'Path isolation is not separate replay/ROI authorization.'});print(json.dumps({'plans':len(rows),'ready':sum(not r['blocking_gates']for r in rows),'staged':sum(r['staged']for r in rows)}))
if __name__=='__main__':main()
