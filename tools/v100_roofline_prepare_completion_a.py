#!/usr/bin/env python3
"""Stage exact reviewed completion-A profiling plans. Never build or execute targets.

Review-bound output-only copies, canonical readonly inputs and fresh scratch;
Landlock remains mandatory. Path readiness is not ROI/replay authorization.
"""
import argparse,copy,difflib,hashlib,json,os,re,shutil,sys
from pathlib import Path
sys.dont_write_bytecode=True

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for block in iter(lambda:f.read(4<<20),b''):h.update(block)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def check(bound):
 for p,h in bound.items():
  if sha(p)!=h:raise ValueError('Binding changed: '+p)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--reserve-mib',type=int,default=2048);a=ap.parse_args();root=a.source_root.resolve();out=a.output_root.resolve();SELF=Path(__file__).resolve();guard=root/'tools/v100_profile_write_guard.py'
 if out.exists()or not out.is_relative_to('/tmp')or out==Path('/tmp')or a.reserve_mib<0:ap.error('Fresh owned /tmp output and nonnegative reserve required')
 if sha(a.review)!=a.review_sha256:raise ValueError('Exact review SHA mismatch')
 review=json.loads(a.review.read_text());check(review['source_bindings']);out.mkdir(parents=True);rows=[]
 for item in review['slots']:
  b,m=item['benchmark'],item['model'];basepath=Path(item['base_plan']);
  if sha(basepath)!=item['base_plan_sha256']:raise ValueError('Base plan drift')
  base=json.loads(basepath.read_text());plan=copy.deepcopy(base);slot=out/b/m;scratch=slot/'scratch';slot.mkdir(parents=True);old=base['scratch_cwd'];move=lambda s:s.replace(old,str(scratch))if isinstance(s,str)else s
  rp=Path(base['reference_result']);cp=rp.with_name('config.json')
  if sha(rp)!=base['reference_result_sha256']or sha(cp)!=base['raw_config_sha256']:raise ValueError('Selected evidence drift')
  check(base['execution_bindings'])
  plan['scratch_cwd']=str(scratch)
  for key in ['build_command','profile_target_command']:plan[key]=[move(v)for v in plan[key]]
  plan['environment_overrides']={k:move(v)for k,v in plan['environment_overrides'].items()}
  for asset in plan['runtime_assets']:asset['destination']=move(asset['destination'])
  bound={**base['execution_bindings'],**review['source_bindings'],str(a.review.resolve()):a.review_sha256,str(SELF):sha(SELF),str(guard):sha(guard),str(basepath):sha(basepath),str(cp):sha(cp)};patches=[];precreate=[];cleared=[];mapping={}
  def patch(src,text,label):
   src=Path(src);original=src.read_text()
   if review['source_bindings'].get(str(src))!=sha(src):raise ValueError('Patch source absent exact review '+str(src))
   dest=scratch/'profile_source'/label;mapping[str(src)]=str(dest)
   patches.append({'original':str(src),'original_sha256':sha(src),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'unified_diff':''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(src),tofile=str(dest)))})
   return dest
  def replace(text,oldtext,newtext):
   if text.count(oldtext)!=1:raise ValueError('Exact replacement count mismatch '+oldtext)
   return text.replace(oldtext,newtext)
  def native_text(src,text):
   def fix(match):
    p=(src.parent/match[2]).resolve()
    if not p.is_file():raise ValueError('Missing canonical quoted include '+str(p))
    bound[str(p)]=sha(p);return match[1]+json.dumps(str(p))
   return re.sub(r'(^\s*#\s*include\s*)"([^"]+)"',fix,text,flags=re.M)
  def julia_text(src,text):
   # Pin original canonical @__DIR__ for all unchanged includes/read-only fixtures.
   text=text.replace('@__DIR__',json.dumps(str(src.parent)))
   def fix(match):
    p=(src.parent/match[1]).resolve()
    if not p.is_file():raise ValueError('Missing canonical Julia include '+str(p))
    bound[str(p)]=sha(p);return 'include('+json.dumps(str(p))+')'
   return re.sub(r'include\("([^"\n]+)"\)',fix,text)
  case=item['case'];srcroot=root/'src'/(b+'-julia')
  if case in ['ccs_native_tmpfile','bn_native_failure_capture','clock_native_outputs','memcpy_native_output']:
   f=Path(item['source']);text=f.read_text()
   for r in item['replacements']:text=replace(text,r['old'],r['new'].replace('${OUTPUT}',str(scratch/'profile_outputs')).replace('${SCRATCH}',str(scratch)))
   dest=patch(f,native_text(f,text),f.name)
   if plan['build_command'].count(str(f))!=1:raise ValueError('Unique native compiler source argument required')
   plan['build_command']=[str(dest)if x==str(f)else x for x in plan['build_command']]
  elif case=='cm_jacc_trace':
   f=Path(item['source']);text=f.read_text()
   for r in item['replacements']:text=replace(text,r['old'],r['new'].replace('${OUTPUT}',str(scratch/'profile_outputs')))
   dest=patch(f,julia_text(f,text),f.name);plan['profile_target_command']=[str(dest)if x==str(f)else x for x in plan['profile_target_command']]
  elif case=='miniDGS_persist':
   f=srcroot/'study_driver.py';text=f.read_text()
   for r in item['replacements']:text=replace(text,r['old'],r['new'])
   dest=patch(f,text,f.name);plan['profile_target_command']=[str(dest)if x==str(f)else x for x in plan['profile_target_command']]
   if m=='cuda':
    # Original wrapper resolves the native rank executable under STUDY_OUTPUT_DIR.
    output_index=plan['build_command'].index('-o')+1;expected=scratch/'profile_outputs/native_rank'
    if Path(plan['build_command'][output_index]).name!='native_rank':raise ValueError('Unexpected native rank output')
    plan['build_command'][output_index]=str(expected)
  elif case=='mf_sgd_native_precreate':
   p=Path(plan['profile_target_command'][4]);p.relative_to(scratch);precreate.append(p)
  elif case=='gc_outputs':
   f=srcroot/('study_cuda.cu'if m=='cuda'else'study_common.jl');text=replace(f.read_text(),'/home/imo/HeCBench_julia/.porting-state/portable_julia/raw_sidecars/gc',str(scratch/'profile_outputs/gc_sidecars'))
   dest=patch(f,native_text(f,text)if m=='cuda'else julia_text(f,text),f.name)
   if m=='cuda':plan['build_command']=[str(dest)if x==str(f)else x for x in plan['build_command']]
   else:
    entry=next(Path(x)for x in plan['profile_target_command']if x.endswith('.jl'));entrytext=julia_text(entry,entry.read_text());pattern=r'include\(joinpath\('+re.escape(json.dumps(str(entry.parent)))+r',"\.\.","study_common\.jl"\)\)'
    entrytext,n=re.subn(pattern,'include('+json.dumps(str(dest))+')',entrytext)
    if n!=1:raise ValueError('Unique copied common include required')
    newentry=patch(entry,entrytext,'entry_'+entry.name);plan['profile_target_command']=[str(newentry)if x==str(entry)else x for x in plan['profile_target_command']]
  elif case=='columnar_native_compiler':
   f=srcroot/'run_native.py';text=f.read_text();text=replace(text,'P=Path(__file__).resolve().parent','P=Path('+repr(str(srcroot))+')');text=replace(text,"root=Path('/tmp/columnarSolver_native')/('checked_'+str(os.getpid()))",'root=Path('+repr(str(scratch/'native_captures'))+')');text=replace(text,"'/tmp/columnarSolver_native/study_native_storage_v2'",repr(str(scratch/'study_native_storage_v2')))
   dest=patch(f,text,f.name);plan['profile_target_command']=[str(dest)if x==str(f)else x for x in plan['profile_target_command']]
   compiler=item['reviewed_compiler_command'];plan['build_command']=[x.replace('${SCRATCH}',str(scratch))for x in compiler]
   if compiler[0]!='/usr/bin/nvcc':raise ValueError('Only exact reviewed existing CUDA12.0 nvcc wrapper permitted')
  else:raise ValueError('Unreviewed case')
  for gate in plan['blocking_gates']:
   if gate not in item.get('clear_gates',[]):continue
   if gate!='source_bound_absolute_path_literals_require_write_isolation_review':raise ValueError('Refuse clearing unrelated gate')
   cleared.append(gate)
  plan['blocking_gates']=[x for x in plan['blocking_gates']if x not in cleared]
  # Preserve all semantic knobs; only original scratch prefix and reviewed output roots move.
  compiler=list(plan['build_command'])
  if compiler:
   plan['build_command']=[sys.executable,str(guard),'--allow-dir',str(scratch),'--',*compiler]
   dump(slot/'compiler_manifest.json',{'compiler_command':compiler,'guarded_command':plan['build_command'],'guard_sha256':sha(guard),'compiler_runs':0,'GPU_runs':0});bound[str(slot/'compiler_manifest.json')]=sha(slot/'compiler_manifest.json')
  plan['execution_bindings']=bound;plan['staged']=False;plan['staging_status']='blocked'if plan['blocking_gates']else'input_plan_ready_ROI_and_kernel_mapping_still_required';plan['profiler_requirements']['filesystem_write_guard_required']=True
  context={}
  for directory in [root/'src'/(b+'-julia'),root/'src'/(b+'-cuda')]:
   for p in directory.rglob('*'):
    if p.is_file()and p.suffix in ['.jl','.py','.cu','.cpp','.c','.cuh','.h','.hpp','.inc','.toml']:context[str(p.resolve())]=sha(p)
  plan['review']={'path':str(a.review.resolve()),'sha256':a.review_sha256,'entry':{'case':case,'reason':item['reason'],'cleared_gates':cleared,'additional_current_readonly_input_bindings':context,'kernel_math_changes':False,'profile_only':True}}
  plan['completion_overlay_provenance']={'base_plan':str(basepath),'base_plan_sha256':sha(basepath),'helper_sha256':sha(SELF),'review_sha256':a.review_sha256,'original_config_sha256':base['raw_config_sha256'],'original_result_sha256':base['reference_result_sha256'],'patches':patches,'source_before':{**review['source_bindings'],**context},'source_after':None,'precreated_directories':[str(p)for p in precreate],'compiler_runs':0,'GPU_runs':0,'original_environment':base['environment_overrides'],'new_environment':plan['environment_overrides']}
  if not plan['blocking_gates']:
   if shutil.disk_usage(out).free<plan['runtime_asset_bytes']+(a.reserve_mib<<20):raise ValueError('Storage reserve prevents staging')
   scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir()
   for asset in plan['runtime_assets']:
    src=Path(asset['source']);dest=Path(asset['destination']);dest.relative_to(scratch)
    for q in asset['source_files']:
     if sha(q['path'])!=q['sha256']:raise ValueError('Input asset changed')
    dest.parent.mkdir(parents=True,exist_ok=True)
    if src.is_dir():shutil.copytree(src,dest)
    else:shutil.copy2(src,dest)
    for q in [dest]if dest.is_file()else[p for p in dest.rglob('*')if p.is_file()]:q.chmod(0o555 if os.access(q,os.X_OK)else 0o444)
   for q in patches:
    dest=Path(q['copy']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(q['text']);dest.chmod(0o444);bound[str(dest)]=sha(dest)
    if sha(dest)!=q['copy_sha256']:raise ValueError('Copy mismatch')
   for p in precreate:p.mkdir(parents=True,exist_ok=True)
   check(bound);check(context);plan['completion_overlay_provenance']['source_after']=dict(plan['completion_overlay_provenance']['source_before']);plan['completion_overlay_provenance']['source_bytes_unchanged_after_staging']=True;plan['staged']=True
  dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in plan.items()if base.get(k)!=v});dump(slot/'completion_source_proof.json',plan['completion_overlay_provenance']);bound[str(slot/'completion_source_proof.json')]=sha(slot/'completion_source_proof.json');dump(slot/'plan.json',plan);rows.append({'benchmark':b,'model':m,'plan':str(slot/'plan.json'),'staged':plan['staged'],'blocking_gates':plan['blocking_gates'],'patches':len(patches)})
 dump(out/'summary.json',{'slots':rows,'staged':sum(x['staged']for x in rows),'ready':sum(not x['blocking_gates']for x in rows),'helper_sha256':sha(SELF),'review_sha256':a.review_sha256,'GPU_runs':0,'compiler_runs':0,'source_edits':False,'scope':'Output isolation only; counter coverage/ROI/replay validity remain separately qualified.'});print(json.dumps({'plans':len(rows),'staged':sum(x['staged']for x in rows)}))
if __name__=='__main__':main()
