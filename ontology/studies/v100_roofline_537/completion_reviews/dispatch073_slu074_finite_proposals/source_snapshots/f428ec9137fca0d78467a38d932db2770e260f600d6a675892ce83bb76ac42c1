#!/usr/bin/env python3
"""Exact SLU/SVD3x3 profiling-isolation plans. Default: CPU plans only.

--stage copies declared assets/overlay text to a fresh /tmp tree, never builds.
--build-manifest is compiler-only, reserved for coordinator execution; no target run.
Accepted sources, old caches, original configurations and results remain immutable.
"""
import argparse,copy,difflib,hashlib,json,os,pathlib,re,shutil,subprocess,sys
sys.dont_write_bytecode=True
P=pathlib.Path
CASES={'slu','svd3x3'}
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):P(p).write_text(json.dumps(d,indent=2)+'\n')
def under(p,r):return P(p).resolve().is_relative_to(P(r).resolve())
def check(bindings):
 for p,h in bindings.items():
  if not P(p).is_file()or sha(p)!=h:raise ValueError('Input drift: '+p)
def build_only(path,expected):
 if sha(path)!=expected:raise ValueError('Manifest hash mismatch')
 d=json.loads(path.read_text());scratch=P(d['scratch']).resolve()
 if not d.get('compiler_only')or not under(scratch,'/tmp')or scratch==P('/tmp')or P.cwd().resolve()!=scratch:raise ValueError('Fresh isolated compiler cwd required')
 check(d['input_bindings']);receipt=scratch/'extra_build_outcome.json'
 if receipt.exists():raise ValueError('Refuse repeated build')
 for p in d['must_be_absent_before_build']:
  if P(p).exists()or P(p).is_symlink():raise ValueError('Existing output: '+p)
 result={'GPU_target_executed':False,'source_before':d['input_bindings'],'compiler_commands':[]}
 try:
  for cmd in d['commands']:
   exe=P(cmd[0]).name
   if exe not in {'gcc','g++','nvcc','ar'}:raise ValueError('Not an allowed compiler/archive command')
   dest=cmd[2]if exe=='ar'else cmd[cmd.index('-o')+1]
   if not under(dest,scratch):raise ValueError('Compiler output escapes scratch')
   P(dest).parent.mkdir(parents=True,exist_ok=True)
   result['compiler_commands'].append(cmd);subprocess.run(cmd,cwd=scratch,check=True)
  result['outputs']=[{'path':p,'sha256':sha(p)}for p in d['final_outputs']];result['status']='compiled_only_no_target_execution'
 finally:
  check(d['input_bindings']);result['source_bytes_unchanged']=True;dump(receipt,result)
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);a.add_argument('--base-plan-root',type=P);a.add_argument('--output-root',type=P);a.add_argument('--benchmark',choices=sorted(CASES),action='append');a.add_argument('--models',nargs='+',choices=['cuda','cuda_jl','ka','jacc'],default=['cuda','cuda_jl','ka','jacc']);a.add_argument('--stage',action='store_true');a.add_argument('--asset-cap-mib',type=int,default=512);a.add_argument('--reserve-mib',type=int,default=2048);a.add_argument('--build-manifest',type=P);a.add_argument('--manifest-sha256');z=a.parse_args()
 if z.build_manifest:
  if not z.manifest_sha256:a.error('Exact manifest SHA required')
  build_only(z.build_manifest,z.manifest_sha256);return
 if not z.base_plan_root or not z.output_root:a.error('Base plan and fresh output root required')
 if min(z.asset_cap_mib,z.reserve_mib)<0:a.error('Nonnegative caps required')
 root=z.source_root.resolve();out=z.output_root.absolute();script=P(__file__).resolve()
 if not under(out,'/tmp')or out.resolve()==P('/tmp')or out.exists()or out.is_symlink():a.error('Output must be fresh strictly inside /tmp')
 out.mkdir(parents=True);summary=[]
 for b in z.benchmark or sorted(CASES):
  for m in z.models:
   if b=='svd3x3'and m=='cuda':continue
   basepath=z.base_plan_root/b/m/'plan.json'
   if not basepath.exists():continue
   base=json.loads(basepath.read_text());plan=copy.deepcopy(base);slot=out/b/m;slot.mkdir(parents=True);scratch=slot/'scratch';old=base['scratch_cwd'];bindings=dict(base['execution_bindings']);check(bindings)
   bindings[str(script)]=sha(script);bindings[str(basepath.resolve())]=sha(basepath)
   rp=P(base['reference_result']);
   if sha(rp)!=base['reference_result_sha256']or sha(rp.with_name('config.json'))!=base['raw_config_sha256']:raise ValueError('Reference/config drift')
   def move(x):return x.replace(old,str(scratch))if isinstance(x,str)else x
   plan['scratch_cwd']=str(scratch);plan['environment_overrides']={k:move(v)for k,v in plan['environment_overrides'].items()};plan['build_command']=[move(x)for x in plan['build_command']];plan['profile_target_command']=[move(x)for x in plan['profile_target_command']]
   for asset in plan['runtime_assets']:asset['destination']=move(asset['destination'])
   patches=[];additional={};commands=[];outputs=[];notes=[];clear=[];readonly={}
   def read_bound(p):p=P(p);additional[str(p)]=sha(p);return p.read_text()
   def patch(src,dest,pairs):
    original=read_bound(src);text=original
    for before,after in pairs:
     if text.count(before)!=1:raise ValueError('Exact patch occurrence mismatch '+str(src)+' '+before)
     text=text.replace(before,after)
    patches.append({'source':str(src),'source_sha256':sha(src),'copy':str(dest),'copy_sha256':hashlib.sha256(text.encode()).hexdigest(),'text':text,'diff':''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=str(src),tofile=str(dest)))})
   if b=='slu':
    S=root/'src/slu-julia';V=P('/tmp/hecbench-stencil-ports-slu-original/src/nicslu');make=read_bound(V/'lib/Makefile');objs=re.findall(r'([A-Za-z0-9_]+)\.o',make)
    if len(objs)!=len(set(objs))or not objs:raise ValueError('NICSLU object list changed')
    for directory in [V/'source',V/'include',V/'util']:
     for p in directory.rglob('*'):
      if p.is_file()and p.suffix in {'.c','.h'}:additional[str(p)]=sha(p)
    cache=scratch/'private_nicslu';objects=[]
    for name in objs:
     obj=cache/(name+'.o');objects.append(str(obj));commands.append(['gcc','-fPIC','-g','-O2','-msse2','-DNO_ATOMIC','-DSSE2','-c',str(V/'source'/(name+'.c')),'-o',str(obj)])
    commands.append(['ar','rcs',str(cache/'nicslu.a'),*objects]);u=cache/'nicslu_readfile.o';commands.append(['gcc','-fPIC','-g','-O2','-msse2','-DNO_ATOMIC','-DSSE2','-I'+str(V/'include'),'-c',str(V/'util/nicslu_readfile.c'),'-o',str(u)]);commands.append(['ar','rcs',str(cache/'nicslu_util.a'),str(u)])
    target=scratch/('study_native'if m=='cuda'else'study_gpu.so');commands.append(['/usr/bin/nvcc','-std=c++17','-O3','-arch=sm_70','-DVERIFY','-I'+str(S/'reference_history/include'),'-I'+str(V/'include'),'-I'+str(V/'util'),str(S/'full_study.cu'),str(S/'checked_preprocess.cpp'),str(S/'reference_history/src/symbolic.cpp'),str(S/'reference_history/src/Timer.cpp'),str(cache/'nicslu.a'),str(cache/'nicslu_util.a'),'-Xcompiler=-fopenmp','-Xcompiler=-pthread','-lz','-lcrypto','-o',str(target),*(['-DSTUDY_SHARED','--shared','-Xcompiler=-fPIC']if m!='cuda'else[])])
    cmd=plan['profile_target_command'];fixture=next(x for x in cmd if x=='/tmp/hecbench-stencil-ports-slu-fixtures-v2');readonly[fixture]='input_directory'
    fp=P(fixture);additional[str(fp/'cases.txt')]=sha(fp/'cases.txt');additional[str(fp/'canonical_actual_input_prefix.bin')]=sha(fp/'canonical_actual_input_prefix.bin')
    for line in(fp/'cases.txt').read_text().splitlines():
     path=P(line.split('\t',2)[2]);additional[str(path)]=sha(path)
    if m=='cuda':
     if cmd[-1]!='0':raise ValueError('Native prepare must stay0')
     cmd[0]=str(target);private_index=2
    else:
     entry=next(x for x in cmd if x.endswith('/main.jl'));private_index=cmd.index(entry)+3
     if cmd[cmd.index(entry)+1]!='study_gpu.so':raise ValueError('Unexpected SLU library arg')
     cmd[cmd.index(entry)+1]=str(target)
     common=read_bound(S/'study_common.jl')
     if ',abspath(ARGS[2]),abspath(ARGS[3]),0)'not in common:raise ValueError('Julia prepare0 proof changed')
    old_private=cmd[private_index];cmd[private_index]=str(scratch/'private_trace')
    clear=[g for g in base['blocking_gates']if g in ['custom_build_script_requires_explicit_output_and_side_effect_review','source_bound_absolute_path_literals_require_write_isolation_review','absolute_directory_argument_requires_role:'+fixture,'absolute_directory_argument_requires_role:'+old_private]]
    outputs=[str(target),str(cache/'nicslu.a'),str(cache/'nicslu_util.a')];notes=['Old build_program.py/cache/lock are never invoked. Same Makefile object list and GCC PIC/g/O2/msse2/NO_ATOMIC/SSE2 commands; only archive/object destinations fresh private_nicslu. No shared-cache manifest mutation.','Same original nvcc source/flags; fresh native/shared output. Exact fixture inputs read only with prepare0 (nativeargv/JuliaCABI); private captures redirected, output_trialN.bin and x.dat cwd remain scratch. CPU symbolic/solve andGPUalgorithm/ROI unchanged.']
   else:
    S=root/'src/svd3x3-julia';cmd=plan['profile_target_command'];builder=plan['build_command'];oldlib=str(S/'study_check.so');target=scratch/'study_check.so'
    if builder[builder.index('-o')+1]!=oldlib:raise ValueError('SVD original checker output changed')
    builder[builder.index('-o')+1]=str(target);commands=[builder];outputs=[str(target)]
    if cmd[:1]!=['env']or not cmd[1].startswith('STUDY_RAW_DIR=')or P(cmd[2]).name!='julia':raise ValueError('SVD env executable prefix changed')
    exe=P(cmd[2]);additional[str(exe)]=sha(exe);cmd=cmd[2:];plan['environment_overrides']['STUDY_RAW_DIR']=str(scratch/'private_raw')
    common=S/'study_common.jl';newcommon=scratch/'profile_source/study_common.jl';patch(common,newcommon,[('const PARENT=normpath(joinpath(@__DIR__))','const PARENT='+json.dumps(str(S))),('const CHECK_LIB=joinpath(PARENT,"study_check.so")','const CHECK_LIB=joinpath(pwd(),"study_check.so")')])
    entry=P(next(x for x in cmd if x.endswith('/main.jl')));newentry=scratch/'profile_source/main.jl';patch(entry,newentry,[('include("../device_common.jl")','include('+json.dumps(str(S/'device_common.jl'))+')'),('include("../study_common.jl")','include('+json.dumps(str(newcommon))+')')]);cmd[cmd.index(str(entry))]=str(newentry);plan['profile_target_command']=cmd
    clear=[g for g in base['blocking_gates']if g in ['build_output_outside_new_scratch','unbound_absolute_file_argument_requires_role:'+str(exe)]];notes=['Same g++ checker source/flags, -o only redirected to fresh scratch. Patched common retains canonical source PARENT read-only assets but loads newly built pwd/study_check.so.','Remove only env wrapper/fixed historical STUDY_RAW_DIR token, preserve exact Julia executable/project args; set private_raw fresh environment. Entry includes bound canonical device_common and copiedcommon, no device kernel/function body or numerical/iteration/ROI changes.']
   remaining=[g for g in base['blocking_gates']if g not in clear]
   forbidden=('original_contract_requires_multiple_devices','effective_timing_region_missing','runtime_asset_destination_escape','missing_runtime_asset','runtime_asset_copy_budget')
   if any(g.startswith(forbidden)for g in clear):raise ValueError('Forbidden gate removal')
   bytes_=plan['runtime_asset_bytes']
   if bytes_>z.asset_cap_mib*1024**2:remaining.append('extra_asset_copy_budget_exceeded')
   manifest={'compiler_only':True,'scratch':str(scratch),'commands':commands,'final_outputs':outputs,'must_be_absent_before_build':[cmd[2]if P(cmd[0]).name=='ar'else cmd[cmd.index('-o')+1]for cmd in commands],'input_bindings':{**bindings,**additional}}
   mp=slot/'build_manifest.json';dump(mp,manifest);bindings[str(mp)]=sha(mp)
   plan['build_command']=[sys.executable,str(script),'--build-manifest',str(mp),'--manifest-sha256',sha(mp)];plan['blocking_gates']=remaining;plan['staging_status']='blocked'if remaining else'input_plan_ready_ROI_and_kernel_mapping_still_required';plan['execution_bindings']={**bindings,**additional}
   plan['review']={'case_specific_extra_overlay':True,'profile_only':True,'kernel_body_changes':False,'source_math_changes':False,'original_accepted_config_immutable':True,'cleared_exact_gates':clear,'readonly_path_roles':readonly,'reason':notes,'source_result_sha256':base['reference_result_sha256'],'source_config_sha256':base['raw_config_sha256']}
   plan['overlay_provenance']={'planner_sha256':sha(script),'base_plan_path':str(basepath),'base_plan_sha256':sha(basepath),'additional_current_readonly_bindings':additional,'profile_source_patches':patches,'source_after':None,'compiler_commands_executed_by_planner':False,'GPU_target_executed':False};plan['notes']+=notes;plan['staged']=False
   if z.stage and not remaining:
    if shutil.disk_usage(out).free<bytes_+z.reserve_mib*1024**2:raise ValueError('Free-space reserve prevents stage')
    scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir()
    for asset in plan['runtime_assets']:
     src=P(asset['source']);dst=P(asset['destination'])
     for item in asset['source_files']:
      if sha(item['path'])!=item['sha256']:raise ValueError('Asset drift')
     dst.parent.mkdir(parents=True,exist_ok=True)
     if src.is_dir():shutil.copytree(src,dst)
     else:shutil.copy2(src,dst)
     for f in([dst]if dst.is_file()else[x for x in dst.rglob('*')if x.is_file()]):f.chmod(0o555 if os.access(f,os.X_OK)else 0o444)
    for patch_ in patches:
     dest=P(patch_['copy']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(patch_['text']);dest.chmod(0o444)
     if sha(dest)!=patch_['copy_sha256']:raise ValueError('Overlay mismatch')
     plan['execution_bindings'][str(dest)]=patch_['copy_sha256']
    check(bindings);check(additional);plan['overlay_provenance']['source_after']=dict(additional);plan['staged']=True
   dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in plan.items()if base.get(k)!=v});dump(slot/'plan.json',plan);summary.append({'benchmark':b,'model':m,'blocking_gates':remaining,'staged':plan['staged'],'plan':str(slot/'plan.json')})
 dump(out/'summary.json',{'models':summary,'GPU_executions':0,'compiler_executions':0,'shared_cache_mutations':False,'planner_sha256':sha(script),'ROI_qualification':'Input/build isolation only; no executed-kernel/counter/replay/fullROI claim.'});print(json.dumps({'models':len(summary),'ready':sum(not r['blocking_gates']for r in summary),'summary':str(out/'summary.json')}))
if __name__=='__main__':main()
