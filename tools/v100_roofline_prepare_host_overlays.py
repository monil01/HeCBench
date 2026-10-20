#!/usr/bin/env python3
"""Prepare exact host/output-path-only profiling overlays; never compile or run.

Uses immutable selected plans plus hash-pinned proposal/role reviews. Only new
/tmp outputs may be written. Builds remain collector commands under Landlock.
Replay safety and timed-ROI/kernel coverage require independent review.
"""
import argparse,copy,difflib,hashlib,json,os,pathlib,re,shutil,sys
sys.dont_write_bytecode=True
P=pathlib.Path
SLOTS=[(b,m)for b in ('hungarian','minibude','sparkler')for m in ('cuda_jl','ka','jacc')]+[('columnarSolver','cuda'),('stencil1d','cuda')]
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def under(p,r):
 try:P(p).resolve().relative_to(P(r).resolve());return True
 except ValueError:return False
def check(bindings):
 for p,h in bindings.items():
  if not P(p).is_file()or sha(p)!=h:raise ValueError('Source/input drift: '+p)
def dump(p,x):P(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);a.add_argument('--base-plan-root',type=P,required=True);a.add_argument('--proposals',type=P,required=True);a.add_argument('--proposals-sha256',required=True);a.add_argument('--roles',type=P,required=True);a.add_argument('--roles-sha256',required=True);a.add_argument('--output-root',type=P,required=True);a.add_argument('--benchmark',action='append',choices=sorted({x[0]for x in SLOTS}));a.add_argument('--stage',action='store_true');a.add_argument('--asset-cap-mib',type=int,default=512);a.add_argument('--reserve-mib',type=int,default=2048);z=a.parse_args();R=z.source_root.resolve();O=z.output_root.absolute();SELF=P(__file__).resolve();G=R/'tools/v100_profile_write_guard.py'
 if not under(O,'/tmp')or O.resolve()==P('/tmp')or O.exists()or O.is_symlink():a.error('Fresh output root strictly under /tmp required')
 if min(z.asset_cap_mib,z.reserve_mib)<0:a.error('Nonnegative budgets required')
 for p,h in [(z.proposals,z.proposals_sha256),(z.roles,z.roles_sha256)]:
  if sha(p)!=h:raise ValueError('Review document hash mismatch')
 proposals={x['slot']:x for x in json.loads(z.proposals.read_text())['proposals']};roles={x['slot']:x for x in json.loads(z.roles.read_text())['models']};O.mkdir(parents=True);rows=[]
 for b,m in SLOTS:
  if z.benchmark and b not in z.benchmark:continue
  key=b+'/'+m;f=z.base_plan_root/b/m/'plan.json';base=json.loads(f.read_text());p=copy.deepcopy(base);slot=O/b/m;S=slot/'scratch';old=base['scratch_cwd'];slot.mkdir(parents=True)
  documents=[proposals[key]]if key in proposals else[]
  if key in roles:documents.append(roles[key])
  if b!='hungarian'and not documents:raise ValueError('Missing case review '+key)
  for d in documents:
   if (d['source_result_sha256']!=base['reference_result_sha256']or d['source_config_sha256']!=base['raw_config_sha256']or d['execution_bindings']!=base['execution_bindings']):raise ValueError('Review join mismatch '+key)
  rp=P(base['reference_result']);cp=rp.with_name('config.json')
  if sha(rp)!=base['reference_result_sha256']or sha(cp)!=base['raw_config_sha256']:raise ValueError('Accepted evidence drift')
  check(base['execution_bindings']);before=dict(base['execution_bindings']);bindings={**before,str(SELF):sha(SELF),str(G):sha(G),str(cp):sha(cp),str(f.resolve()):sha(f),str(z.proposals.resolve()):z.proposals_sha256,str(z.roles.resolve()):z.roles_sha256};extra={};patches=[];precreate=[];cleared=[];notes=[]
  for directory in [R/'src'/(b+'-julia'),R/'src'/(b+'-cuda')]:
   for x in directory.rglob('*'):
    if x.is_file()and x.suffix in {'.jl','.cu','.cuh','.cpp','.c','.cc','.h','.hpp','.py','.inc','.toml'}:extra[str(x.resolve())]=sha(x)
  def move(x):return x.replace(old,str(S))if isinstance(x,str)else x
  for k in ['profile_target_command','build_command']:p[k]=[move(x)for x in p[k]]
  p['scratch_cwd']=str(S);p['environment_overrides']={k:move(v)for k,v in p['environment_overrides'].items()}
  for asset in p['runtime_assets']:asset['destination']=move(asset['destination'])
  def patch(source,dest,changes):
   text=source.read_text();original=text
   if before.get(str(source.resolve()))!=sha(source):raise ValueError('Patch source not bound by selected model: '+str(source))
   for x,y in changes:
    if text.count(x)!=1:raise ValueError('Exact unique fragment required: '+str(source))
    text=text.replace(x,y)
   record={'original':str(source.resolve()),'original_sha256':sha(source),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'changes':[{'before':x,'after':y}for x,y in changes],'unified_diff':list(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(source),tofile=str(dest)))};patches.append(record);return dest
  def common_entry(source,changes):
   common=S/'profile_source/study_common.jl';patchedtext=source.read_text()
   for x,y in changes:patchedtext=patchedtext.replace(x,y)
   # Canonicalize directory lookup only; never rewrite arithmetic/device code.
   for line in patchedtext.splitlines():
    if '@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(source.parent)))))
   patch(source,common,changes)
   candidates=[x for x in p['profile_target_command']if x.endswith('/main.jl')]
   if len(candidates)!=1:raise ValueError('Unique model entry required')
   entry=P(candidates[0]);new=S/'profile_source/main.jl';lines=entry.read_text().splitlines();includes=[x for x in lines if x.strip().startswith('include(')and'study_common.jl'in x]
   if len(includes)!=1:raise ValueError('Exact common include missing')
   changes=[(includes[0],'include('+json.dumps(str(common))+')')]
   for line in lines:
    if line!=includes[0]and'@__DIR__'in line:changes.append((line,line.replace('@__DIR__',json.dumps(str(entry.parent)))))
   # Any remaining literal relative include must retain canonical source lookup.
   for line in lines:
    match=re.fullmatch(r'include\("([^"]+)"\)',line.strip())
    if line!=includes[0]and match:changes.append((line,'include('+json.dumps(str((entry.parent/match[1]).resolve()))+')'))
   patch(entry,new,changes);p['profile_target_command']=[str(new)if x==str(entry)else x for x in p['profile_target_command']]
  if b=='hungarian':
   source=R/'src/hungarian-julia/study_common.jl';oldroot='auxdir=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","hungarian-julia-$(H_MODEL)","aux_pid$(getpid())");mkpath(auxdir)'
   common_entry(source,[(oldroot,'auxdir=joinpath(ENV["STUDY_OUTPUT_DIR"],"hungarian_private_raw");mkpath(auxdir)')]);notes.append('Only raw capture root and canonical include/fixture lookups changed; managed control flags, iteration stages and kernels unchanged.')
  elif b=='minibude':
   raw=[(i,x)for i,x in enumerate(p['profile_target_command'])if x.startswith('STUDY_RAW_DIR=')]
   expected='STUDY_RAW_DIR='+str(R/'.porting-state/porting_logs/minibude-julia-shared/raw')
   if len(raw)!=1 or raw[0][1]!=expected:raise ValueError('Unexpected minibude raw assignment')
   p['profile_target_command'][raw[0][0]]='STUDY_RAW_DIR='+str(S/'profile_raw');precreate.append(str(S/'profile_raw'));notes.append('Only inline raw output assignment redirected; selected canonical entry/kernel/common sources and other env values untouched.')
  elif b=='sparkler':
   source=R/'src/sparkler-julia/study_common.jl';common_entry(source,[('capture=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/sparkler_primary"))','capture=joinpath(ENV["STUDY_OUTPUT_DIR"],"sparkler_private_raw")')]);notes.append('ROOT remains canonical source; LIB remains fresh cwd study_backend.so. MPI4 same-GPU6, streams, collectives and all math unchanged. Backend CAS/ledgers follow explicit capture root, primary trial files are cwd-relative. Replay/collective progress remains a separate review.')
  elif b=='columnarSolver':
   source=R/'src/columnarSolver-julia/run_native.py';dest=S/'profile_source/run_native.py'
   changes=[("P=Path(__file__).resolve().parent;root=Path('/tmp/columnarSolver_native')/('checked_'+str(os.getpid()));",'P=Path('+repr(str(source.parent))+');root=Path(os.environ[\'STUDY_OUTPUT_DIR\'])/\'columnar_native_private\';'),("'/tmp/columnarSolver_native/study_native_storage_v2'",repr(str(S/'study_native_storage_v2')))]
   patch(source,dest,changes)
   if p['profile_target_command']!=['python3',str(source)]:raise ValueError('Unexpected columnar wrapper argv')
   p['profile_target_command'][1]=str(dest)
   p['build_command']=['/usr/local/cuda/bin/nvcc','-O3','-std=c++17','-arch=sm_70','-DDATA_PATH='+json.dumps(str(R/'src/columnarSolver-cuda/data')),str(source.parent/'study_native_storage_v2.cu'),'-o',str(S/'study_native_storage_v2')]
   notes.append('Fresh build follows accepted README CUDA12.0 command/flags and exact DATA_PATH macro. Old binary never executed/overwritten; original CPUgold argument remains readonly. Rawroot mkdir remains exclusive.')
  elif b=='stencil1d':
   source=R/'src/stencil1d-julia/study_cuda.cu';dest=S/'profile_source/study_cuda.cu';patch(source,dest,[('char artifact[]="/tmp/hecbench-stencil1d-XXXXXX";','char artifact[]='+json.dumps(str(S/'hecbench-stencil1d-XXXXXX'))+';'),('#include "../stencil1d-cuda/stencil_1d.cu"','#include '+json.dumps(str(R/'src/stencil1d-cuda/stencil_1d.cu')))])
   if p['build_command'].count(str(source))!=1:raise ValueError('Unexpected stencil source argv')
   p['build_command']=[str(dest)if x==str(source)else x for x in p['build_command']];notes.append('mkstemp exclusive creation and all captured output bytes preserved; only template path and canonical include lookup changed. Originaltemp files untouched.')
  if key in roles:
   role=roles[key]['reviewed_readonly_role'];exe=role['argument'];record=role['path_binding']
   if exe not in p['profile_target_command']or sha(exe)!=record['sha256']or str(P(exe).resolve())!=role['resolved_path']or not os.access(exe,os.X_OK):raise ValueError('Readonly executable role drift')
   bindings[exe]=record['sha256'];bindings[str(P(exe).resolve())]=record['sha256'];cleared.append('unbound_absolute_file_argument_requires_role:'+exe);notes.append('Nested Julia executable unchanged, readonly symlink/realpath/hash bound; no Juliaup channel/version change.')
  if b in {'columnarSolver','stencil1d'}:cleared.append('source_bound_absolute_path_literals_require_write_isolation_review')
  # No other gate is cleared, regardless of successful write confinement.
  p['blocking_gates']=[g for g in base['blocking_gates']if g not in cleared]
  assets=p['runtime_asset_bytes']
  if assets>z.asset_cap_mib*1024**2:p['blocking_gates'].append('host_overlay_asset_copy_budget_exceeded')
  if p['build_command']:
   original=p['build_command'];p['build_command']=[sys.executable,str(G),'--allow-dir',str(S),'--',*original];dump(slot/'compiler_manifest.json',{'compiler_command':original,'guarded_build_command':p['build_command'],'guard_sha256':sha(G),'compiler_executed':False,'GPU_executed':False,'reason':'Compiler commands only, no benchmark call; child Landlock restricts writes to fresh scratch.'});bindings[str(slot/'compiler_manifest.json')]=sha(slot/'compiler_manifest.json')
  p['execution_bindings']=bindings;p['staged']=False;p['staging_status']='blocked'if p['blocking_gates']else'input_plan_ready_ROI_and_kernel_mapping_still_required'
  p['review']={'path':str(SELF),'sha256':sha(SELF),'entry':{'reason':notes,'output_isolation_reviewed':True,'additional_current_readonly_input_bindings':extra,'readonly_executable_review':roles.get(key),'source_result_sha256':base['reference_result_sha256'],'source_config_sha256':base['raw_config_sha256'],'only_cleared_gates':[g for g in base['blocking_gates']if g in cleared],'profile_only':True,'kernel_math_changes':False}}
  p['profiler_requirements']['filesystem_write_guard_required']=True;p['notes']+=notes;p['host_overlay_provenance']={'base_plan_path':str(f.resolve()),'base_plan_sha256':sha(f),'planner_sha256':sha(SELF),'guard_sha256':sha(G),'patches':patches,'source_before':{**before,**extra},'source_after':None,'precreated_directories':precreate,'compiler_runs':0,'GPU_runs':0}
  if z.stage and not p['blocking_gates']:
   if shutil.disk_usage(O).free<assets+z.reserve_mib*1024**2:raise ValueError('Storage reserve prevents staging')
   S.mkdir();(S/'profile_outputs').mkdir();(S/'ncu_tmp').mkdir()
   for asset in p['runtime_assets']:
    src=P(asset['source']);dest=P(asset['destination']);dest.parent.mkdir(parents=True,exist_ok=True)
    for item in asset['source_files']:
     if sha(item['path'])!=item['sha256']:raise ValueError('Runtime input changed')
    if src.is_dir():shutil.copytree(src,dest)
    else:shutil.copy2(src,dest)
    for x in [dest]if dest.is_file()else[q for q in dest.rglob('*')if q.is_file()]:x.chmod(0o555 if os.access(x,os.X_OK)else 0o444)
   for q in patches:
    dest=P(q['copy']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(q['text']);dest.chmod(0o444)
    if sha(dest)!=q['copy_sha256']:raise ValueError('Copy patch write mismatch')
    bindings[str(dest)]=q['copy_sha256']
   for directory in precreate:P(directory).mkdir()
   check(bindings);check(extra);p['host_overlay_provenance']['source_after']=dict(p['host_overlay_provenance']['source_before']);p['host_overlay_provenance']['source_bytes_unchanged_after_staging']=True;p['staged']=True
  dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in p.items()if base.get(k)!=v});dump(slot/'plan.json',p);rows.append({'benchmark':b,'model':m,'plan':str(slot/'plan.json'),'staged':p['staged'],'blocking_gates':p['blocking_gates'],'patch_count':len(patches)})
 dump(O/'summary.json',{'models':rows,'ready':sum(not x['blocking_gates']for x in rows),'staged':sum(x['staged']for x in rows),'planner_sha256':sha(SELF),'compiler_runs':0,'GPU_runs':0,'source_edits':False,'only_new_helper_written_in_source':True,'replay_ROI_qualification':'Separate, not authorized by path readiness.'});print(json.dumps({'plans':len(rows),'ready':sum(not x['blocking_gates']for x in rows),'staged':sum(x['staged']for x in rows)}))
if __name__=='__main__':main()
