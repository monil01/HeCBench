#!/usr/bin/env python3
"""Stage EGS compile-phase isolation + ASSERT full-checker diagnostic retries only.
No subprocess/build/GPU/target runs in this planner. All originals immutable.
"""
import copy,difflib,hashlib,json,os,shutil,sys
from pathlib import Path
D=Path(__file__).resolve().parent;S=Path('/home/imo/HeCBench_julia');BASE=D/'current_base_v7';OUT=Path(sys.argv[1]).resolve();GUARD=S/'tools/v100_profile_write_guard.py';SELF=Path(__file__).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def check(d):
 for p,h in d.items():
  if sha(p)!=h:raise ValueError('Binding drift '+p)
if OUT.exists()or not OUT.is_relative_to('/tmp')or OUT==Path('/tmp'):raise SystemExit('Require fresh owned output root')
OUT.mkdir(parents=True);records=[]
for b,models in [('egs',['cuda','cuda_jl','ka','jacc']),('assert',['cuda_jl','ka','jacc'])]:
 for m in models:
  f=BASE/b/m/'plan.json';base=json.loads(f.read_text());plan=copy.deepcopy(base);slot=OUT/b/m;scratch=slot/'scratch';slot.mkdir(parents=True);scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();(scratch/'profile_source').mkdir();old=base['scratch_cwd'];move=lambda x:x.replace(old,str(scratch))if isinstance(x,str)else x
  if base['blocking_gates']or base['runtime_assets']:raise ValueError('Only reviewed no-assets ready base plans')
  check(base['execution_bindings']);rp=Path(base['reference_result'])
  if sha(rp)!=base['reference_result_sha256']or sha(rp.with_name('config.json'))!=base['raw_config_sha256']:raise ValueError('Contract drift')
  plan['scratch_cwd']=str(scratch)
  for key in ['build_command','profile_target_command']:plan[key]=[move(v)for v in plan[key]]
  plan['environment_overrides']={k:move(v)for k,v in plan['environment_overrides'].items()};bound={**base['execution_bindings'],str(f):sha(f),str(SELF):sha(SELF),str(GUARD):sha(GUARD)};patches=[]
  def patched(original,text,name):
   original=Path(original);target=scratch/'profile_source'/name;before=original.read_text();target.write_text(text);target.chmod(0o444);bound[str(original)]=sha(original);bound[str(target)]=sha(target);patches.append({'original':str(original),'original_sha256':sha(original),'copy':str(target),'copy_sha256':sha(target),'diff':''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile=str(original),tofile=str(target)))});return target
  if b=='egs':
   src=S/'src/egs-julia';checker=src/'check_mt_trajectory.cpp';compiler=Path('/usr/bin/g++').resolve();bound[str(checker)]=sha(checker);bound[str(compiler)]=sha(compiler);bound['/usr/bin/g++']=sha('/usr/bin/g++');commands=[list(plan['build_command']),['/usr/bin/g++','-O3','-std=c++17',str(checker),'-o',str(scratch/'mtcheck')]]
   # Same exact library command, plus exact original CPU checker flags out of NCU.
   if commands[0][0]=='nvcc':commands[0][0]='/usr/bin/nvcc'
   bound['/usr/bin/nvcc']=sha('/usr/bin/nvcc');bound['/usr/lib/nvidia-cuda-toolkit/bin/nvcc']=sha('/usr/lib/nvidia-cuda-toolkit/bin/nvcc')
   inputs={str(checker):sha(checker)}
   for arg in commands[0]:
    p=Path(arg)
    if p.is_file()and p.suffix in ['.cu','.cpp','.cuh','.h']:inputs[str(p.resolve())]=sha(p)
   build_manifest=slot/'egs_build_manifest.json';dump(build_manifest,{'scratch':str(scratch),'commands':commands,'input_bindings':{**base['execution_bindings'],**inputs},'reason':'Unchanged CPU MT checker compilation moved before NCU. No claim exact compiler255 cause established; source CPUchecker and flags unchanged, every actual check still runs.'});bound[str(build_manifest)]=sha(build_manifest)
   launcher=D/'build_egs_checked.py';bound[str(launcher)]=sha(launcher);plan['build_command']=[sys.executable,str(GUARD),'--allow-dir',str(scratch),'--',sys.executable,str(launcher),str(build_manifest),sha(build_manifest)]
   gate=D/'checker_gate.py';bound[str(gate)]=sha(gate)
   if m=='cuda':
    native=src/'study_native.py';text=native.read_text();text=text.replace('from assemble_artifact import assemble','sys.path.insert(0,'+repr(str(src))+')\nfrom assemble_artifact import assemble');text=text.replace('HERE=Path(__file__).resolve().parent','HERE=Path('+repr(str(src))+')');oldcompile="subprocess.run(['g++','-O3','-std=c++17',str(HERE/'check_mt_trajectory.cpp'),'-o','mtcheck'],check=True)"
    if text.count(oldcompile)!=1:raise ValueError('Native checker compilation fragment changed')
    text=text.replace(oldcompile,"assert Path('mtcheck').is_file(), 'Reviewed independently built checker required; no in-profile compilation'")
    copyname=patched(native,text,'study_native.py');plan['profile_target_command']=[str(copyname)if v==str(native)else v for v in plan['profile_target_command']]
   plan['profile_target_command']=[sys.executable,str(gate),*plan['profile_target_command']]
   rationale='Same library/kernels/fixtures/five checks. Independent CPU checker exact source and-O3/-std=c++17 unchanged; compile before NCU and require build-hash gate. Native wrapper only canonical imports + replaces redundant compile by existence assertion. Julia finalizer already skips compile when fresh mtcheck exists. Not a claim exact original exit255 cause is known.'
  else:
   src=S/'src/assert-julia';wrapper=src/'run_study.py';text=wrapper.read_text();first='import subprocess,sys,os,json,hashlib,pathlib\n'
   if not text.startswith(first)or text.count('subprocess.run(')!=2:raise ValueError('Original full assertion checker changed')
   prefix=(D/'assert_capture_prefix.py.txt').read_text().replace('__DIAGNOSTIC_PATH__',repr(str(slot/'assert_child_diagnostics.json')));bound[str(D/'assert_capture_prefix.py.txt')]=sha(D/'assert_capture_prefix.py.txt');rest=text[len(first):].replace('subprocess.run(','_captured_run(');copyname=patched(wrapper,first+prefix+'\n'+rest,'run_study.py')
   plan['profile_target_command']=[str(copyname)if v==str(wrapper)else v for v in plan['profile_target_command']];installed=S.parent/'.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia'
   if plan['profile_target_command'].count('julia')!=1:raise ValueError('Exact Julia launcher argument not unique')
   plan['profile_target_command']=[str(installed)if v=='julia'else v for v in plan['profile_target_command']];bound[str(installed)]=sha(installed)
   # Initialize persistent diagnostics before any future process; outside disposable scratch.
   dump(slot/'assert_child_diagnostics.json',{'scope':'Not executed. Future full raw child bytes and signed status retained before original checker rejection.','processes':[]})
   rationale='Unproven diagnostic recovery: direct exact installedJulia1.12.6 removes version-manager dispatch, not math/sourcekernel changes. All5expected710 isolatednegatives and all5positivePERFchecks/fields retained byte-for-byte in original checker body; only subprocess runner adds raw diagnostics before rejection. No positive-only bypass or acceptance weakening. Negative failingsecond invocation should not be selected by unchanged first-invocation-per-function/geometry filter; actual profiler behavior remains a failure gate.'
  context={}
  for directory in [S/'src'/(b+'-julia'),S/'src'/(b+'-cuda')]:
   for p in directory.rglob('*'):
    if p.is_file()and p.suffix in ['.jl','.py','.cu','.cpp','.c','.h','.cuh','.inc','.toml']:context[str(p.resolve())]=sha(p)
  bound.update(context);plan['execution_bindings']=bound;plan['staged']=True;plan['staging_status']='input_plan_ready_ROI_and_kernel_mapping_still_required';plan['profiler_requirements']['filesystem_write_guard_required']=True
  proof={'helper':str(SELF),'helper_sha256':sha(SELF),'base_plan':str(f),'base_plan_sha256':sha(f),'selected_result_sha256':base['reference_result_sha256'],'selected_config_sha256':base['raw_config_sha256'],'rationale':rationale,'patches':patches,'source_before':{**base['execution_bindings'],**context},'source_after':{**base['execution_bindings'],**context},'source_bytes_unchanged_after_staging':True,'original_environment':base['environment_overrides'],'new_environment':plan['environment_overrides'],'child_compiler_environment_changes':False,'GPU_runs':0,'build_runs':0,'tests_run':0}
  plan['remaining_infra_provenance']=proof;plan['review']={'path':str(SELF),'sha256':sha(SELF),'entry':{'reason':rationale,'additional_current_readonly_input_bindings':context,'kernel_math_changes':False,'profile_only':True,'recovery_unproven_until_actual_success':True}}
  check(bound);dump(slot/'remaining_infra_source_proof.json',proof);bound[str(slot/'remaining_infra_source_proof.json')]=sha(slot/'remaining_infra_source_proof.json');dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in plan.items()if base.get(k)!=v});dump(slot/'plan.json',plan);records.append({'benchmark':b,'model':m,'plan':str(slot/'plan.json'),'plan_sha256':sha(slot/'plan.json'),'staged':True,'rationale':rationale})
dump(OUT/'summary.json',{'records':records,'GPU_runs':0,'build_runs':0,'tests_run':0,'source_changes':False,'status':'Seven unexecuted parent-reviewable infrastructure isolation/diagnostic retry plans; not recovered successes.'});print(json.dumps({'plans':len(records),'root':str(OUT)}))
