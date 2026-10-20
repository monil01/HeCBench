#!/usr/bin/env python3
"""Prepare conservative per-model profiling plans; NEVER build/run/profiling commands.
Writes only a fresh explicitly supplied /tmp directory. ROI/counter collection is
separate. A ready input-staging plan does not mean a roofline is scientifically valid.
"""
import argparse, collections, hashlib, importlib.util, json, os, pathlib, re, shutil, sys
sys.dont_write_bytecode=True
P=pathlib.Path
MODELS=('cuda','cuda_jl','ka','jacc')
SOURCE_SUFFIXES={'.jl','.py','.cu','.cuh','.cpp','.c','.h','.hpp','.inc','.toml'}
def sha(path):
 h=hashlib.sha256()
 with P(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def under(p,root):
 try:p.resolve().relative_to(root.resolve());return True
 except ValueError:return False

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);ap.add_argument('--census',type=P)
 ap.add_argument('--benchmark',action='append');ap.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
 ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--reviews',type=P)
 ap.add_argument('--stage',action='store_true',help='Copy only explicitly declared runtime assets into new scratch; no command execution.')
 ap.add_argument('--asset-cap-mib',type=int,default=64);ap.add_argument('--reserve-mib',type=int,default=2048)
 a=ap.parse_args();root=a.source_root.resolve();out=a.output_root.absolute()
 if not under(out,P('/tmp')) or out.resolve()==P('/tmp'):ap.error('Output must be a fresh owned directory strictly under /tmp')
 if out.exists() or out.is_symlink():ap.error('Refuse existing output root; accepted directories may never be reused')
 if a.asset_cap_mib<0 or a.reserve_mib<0:ap.error('Storage bounds must be nonnegative')
 protocol_path=root/'ontology/studies/v100_roofline_537/protocol.json'
 protocol=json.loads(protocol_path.read_text())if protocol_path.is_file()else{}
 declared_census=protocol.get('source_census',{})
 cp=a.census or(root/declared_census['path']if declared_census.get('path')else root/'2026-PASC-Hecbench-Julia-paper/data/current_537_contract_census.json');cp=cp.resolve();census=json.loads(cp.read_text())
 if not a.census and declared_census.get('sha256')and sha(cp)!=declared_census['sha256']:raise ValueError('Study protocol census hash mismatch')
 review_doc=json.loads(a.reviews.read_text())if a.reviews else {};reviews=review_doc.get('models',{})
 module_path=root/'tools/study_evidence.py';loader=importlib.util.spec_from_file_location('roofline_effective_evidence',module_path);mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
 out.mkdir(parents=True);plans=[];state=[]
 for entry in census['rows']:
  b=entry['benchmark']
  if a.benchmark and b not in a.benchmark:continue
  rp=P(entry['result']);rp=rp if rp.is_absolute()else root/rp
  result=json.loads(rp.read_text());configpath=rp.with_name('config.json');raw=json.loads(configpath.read_text());config=mod.effective_config(rp,raw)
  result_hash=sha(rp);config_hash=sha(configpath)
  if result_hash!=entry['result_sha256']:raise ValueError('Selected result hash mismatch: '+b)
  effective_roi=(result.get('timing_region_metadata_adjudication')or{}).get('effective_timing_region')
  if not effective_roi:
   regions={t.get('timing_region')for v in result['models'].values()for t in v.get('trials',[])}
   effective_roi=next(iter(regions))if len(regions)==1 and None not in regions else None
  for model in a.models:
   spec=config.get('models',{}).get(model)
   if spec is None:continue
   slot=out/b/model;scratch=slot/'scratch';output=scratch/'profile_outputs'
   blocks=[];notes=[];bindings=result['models'][model].get('adapter_file_hashes',{})
   cwd=P(spec.get('cwd',result['models'][model].get('source_path','')));cwd=cwd if cwd.is_absolute()else root/cwd;cwd=cwd.resolve()
   cmd=list(spec['command']);build=list(spec.get('build',[]));assets=[];asset_bytes=0
   if str(config.get('gpu',6))!='6':blocks.append('original_contract_requires_multiple_devices; no silent single_device_rewrite')
   if entry['outcome']=='execution_only':notes.append('execution_only: arithmetic roofline may be not_applicable; do not assign zero-FLOP memory-bound automatically')
   if not effective_roi:blocks.append('effective_timing_region_missing')
   if build and P(build[0]).name not in ('nvcc','g++','gcc','clang','clang++','nvc++'):
    blocks.append('custom_build_script_requires_explicit_output_and_side_effect_review')
   outputs={}
   for i,arg in enumerate(build):
    if arg=='-o' and i+1<len(build):
     target=P(build[i+1])
     if target.is_absolute()or'..'in target.parts:blocks.append('build_output_outside_new_scratch')
     else:outputs[str(target)]=scratch/target
    elif arg.startswith('-o')and len(arg)>2:
     blocks.append('joined_build_output_flag_requires_review')
   for name in spec.get('runtime_assets',[]):
    rel=P(name)
    if rel.is_absolute()or'..'in rel.parts:blocks.append('runtime_asset_destination_escape:'+name);continue
    source=cwd/rel
    if not source.exists():blocks.append('missing_runtime_asset:'+str(source));continue
    files=[source]if source.is_file()else[x for x in source.rglob('*')if x.is_file()]
    # A source symlink is permitted only as a declared read-only input. Copies
    # dereference it into the new scratch tree; no symlink back to source writes.
    sizes=sum(x.stat().st_size for x in files);asset_bytes+=sizes
    assets.append({'source':str(source),'destination':str(scratch/rel),'bytes':sizes,'source_files':[{'path':str(x),'sha256':sha(x)}for x in files]})
   if asset_bytes>a.asset_cap_mib*1024**2:blocks.append('runtime_asset_copy_budget_exceeded')
   # Detect source-bound absolute paths conservatively; a path literal is not
   # proof of a write. Keep findings and require an audited output-isolation
   # declaration instead of guessing which external files a wrapper modifies.
   literals=[]
   for filename,h in bindings.items():
    f=P(filename)
    if f.suffix not in SOURCE_SUFFIXES or not f.is_file()or f.stat().st_size>2*1024*1024:continue
    for n,line in enumerate(f.read_text(errors='replace').splitlines(),1):
     if re.search(r'''["']/(?:tmp|home)/''',line):
      literals.append({'path':str(f),'line':n,'text':line[:600],'sha256':h})
   if literals:blocks.append('source_bound_absolute_path_literals_require_write_isolation_review')
   def mapped(args,is_build=False):
    resolved=[]
    for i,arg in enumerate(args):
     if arg in outputs or(arg.startswith('./')and arg[2:]in outputs):resolved.append(str(outputs.get(arg,outputs.get(arg[2:]))));continue
     if arg=='{arch}':resolved.append('sm_70');continue
     arg=arg.replace('{arch}','sm_70')
     if i and args[i-1]=='-o':resolved.append(arg);continue
     if arg.startswith(('-I','-L'))and len(arg)>2 and not P(arg[2:]).is_absolute():resolved.append(arg[:2]+str((cwd/arg[2:]).resolve()));continue
     if arg.startswith('--project='):
      x=P(arg.split('=',1)[1]);resolved.append('--project='+str(x if x.is_absolute()else(cwd/x).resolve()));continue
     p=P(arg)
     if not p.is_absolute()and(cwd/p).is_file()and(p.suffix in SOURCE_SUFFIXES or arg.startswith('./')):
      resolved.append(str((cwd/p).resolve()));continue
     if p.is_absolute()and not is_build and arg.startswith(('/tmp/','/home/')):
      # Existing source, runtime dependency or library is only a read-only input
      # after explicit binding. Existing directories and undeclared files block.
      if p.is_dir():blocks.append('absolute_directory_argument_requires_role:'+str(p))
      elif p.is_file()and str(p.resolve())not in bindings and p.suffix not in SOURCE_SUFFIXES and i!=0:
       blocks.append('unbound_absolute_file_argument_requires_role:'+str(p))
      elif not p.exists():blocks.append('absolute_nonexistent_argument_requires_output_redirection:'+str(p))
     resolved.append(arg)
    return resolved
   runcommand=mapped(cmd);buildcommand=mapped(build,True)
   review=reviews.get(b+'/'+model);review_applied=None
   if review:
    if review.get('source_result_sha256')!=result_hash or review.get('source_config_sha256')!=config_hash or review.get('execution_bindings')!=bindings:
     blocks.append('review_binding_mismatch')
    elif not review.get('reason')or review.get('output_isolation_reviewed')is not True:
     blocks.append('review_requires_reason_and_output_isolation_attestation')
    else:
     original_cmd=runcommand.copy()
     for change in review.get('argv_replacements',[]):
      i=change['index']
      if not isinstance(i,int)or not 0<=i<len(cmd)or cmd[i]!=change['original']:raise ValueError('Invalid exact argv review '+b+'/'+model)
      dest=change['replacement'].replace('{scratch}',str(scratch))
      if not under(P(dest),scratch):raise ValueError('Output replacement escapes scratch')
      runcommand[i]=dest
     # Only explicitly acknowledged findings may clear; ROI and multidevice
     # gates are intentionally not clearable by this generic staging tool.
     reviewed=set(review.get('acknowledged_gates',[]))
     for gate in reviewed:
      if gate.startswith('absolute_directory_argument_requires_role:'):
       oldpath=gate.split(':',1)[1]
       redirected=any(change.get('original')==oldpath for change in review.get('argv_replacements',[]))
       if not redirected and review.get('readonly_path_roles',{}).get(oldpath)!='input_directory':raise ValueError('Directory role/redirection missing: '+oldpath)
      if gate.startswith('unbound_absolute_file_argument_requires_role:'):
       oldpath=gate.split(':',1)[1]
       if review.get('readonly_path_roles',{}).get(oldpath)not in('input_file','library','executable'):raise ValueError('Read-only file role missing: '+oldpath)
     never=('original_contract_requires_multiple_devices','effective_timing_region_missing','runtime_asset_destination_escape','missing_runtime_asset','runtime_asset_copy_budget','build_output_outside')
     for gate in reviewed:
      if gate not in blocks or gate.startswith(never):raise ValueError('Review cannot clear gate: '+gate)
     blocks=[x for x in blocks if x not in reviewed]
     review_applied={'path':str(a.reviews.resolve()),'sha256':sha(a.reviews),'entry':review,'command_before_redirection':original_cmd}
   env=dict(config.get('env',{}));env.update(spec.get('env',{}));env.update(CUDA_VISIBLE_DEVICES='6',STUDY_OUTPUT_DIR=str(output),TMPDIR=str(scratch/'ncu_tmp'))
   plan={'benchmark':b,'model':model,'configuration_id':config.get('configuration_id'),'reference_result':str(rp),'reference_result_sha256':result_hash,'raw_config_sha256':config_hash,'effective_config_adjudication':config.get('configuration_classification_adjudication'),'outcome':entry['outcome'],'qualification':entry['qualification'],'source_cwd_readonly_inputs':str(cwd),'scratch_cwd':str(scratch),'build_command':buildcommand,'profile_target_command':runcommand,'environment_overrides':env,'runtime_assets':assets,'runtime_asset_bytes':asset_bytes,'execution_bindings':bindings,'source_path_literal_findings':literals,'review':review_applied,'blocking_gates':sorted(set(blocks)),'notes':notes,'staging_status':'blocked'if blocks else'input_plan_ready_ROI_and_kernel_mapping_still_required','effective_original_timing_region':effective_roi,'kernel_ROI_mapping_status':'unreviewed_no_roofline_claim','profiler_requirements':{'target_processes':'all','root_validated_inventory_selector':['--filter-mode','per-launch-config','--kernel-id','::regex:.*:1'],'selector_scope':'First each function/geometry only; not complete timed-ROI coverage.','profile_from_start':'requires_separate_ROI_review','profiling_only':True,'never_use_instrumented_times_as_primary_five_trial_means':True,'lock_path':'/tmp/hecbench-julia-study-gpu.lock','private_TMPDIR_required':True,'clock_control':'none','cache_control':'all','replay_mode':'must_be_reviewed_per_concurrency_contract'},'commands_executed':False,'binding_validation':'Expected execution hashes recorded; full rehash required under GPU lock before/after build and profile, not performed by inventory-only mode.'}
   slot.mkdir(parents=True)
   if a.stage and not blocks:
    for path,expected in bindings.items():
     if not P(path).is_file()or sha(path)!=expected:raise ValueError('Source/input binding drift prevents staging: '+path)
    plan['binding_validation']='All explicit execution bindings rehashed before staging; execution-time lock checks still required.'
    if shutil.disk_usage(out).free<asset_bytes+a.reserve_mib*1024**2:raise RuntimeError('Free-space reserve prevents new staging')
    scratch.mkdir();output.mkdir();(scratch/'ncu_tmp').mkdir()
    for asset in assets:
     src=P(asset['source']);dest=P(asset['destination']);dest.parent.mkdir(parents=True,exist_ok=True)
     if src.is_dir():shutil.copytree(src,dest)
     else:shutil.copy2(src,dest)
     for f in([dest]if dest.is_file()else[x for x in dest.rglob('*')if x.is_file()]):f.chmod(0o555 if os.access(f,os.X_OK)else 0o444)
    plan['staged']=True
   (slot/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');plans.append(plan)
 for b in census.get('no_aggregate_eligible_accepted_contract_names',[]):
  if a.benchmark and b not in a.benchmark:continue
  for m in a.models:state.append({'benchmark':b,'model':m,'status':'native_hardware_prerequisite_unavailable','profile':None,'classification':None,'reason':'No accepted executable native contract; retain existing terminal evidence, no substitute arithmetic.'})
 summary={'accepted_model_plans':len(plans),'terminal_hardware_slots':state,'input_plan_ready_count':sum(not x['blocking_gates']for x in plans),'blocked_count':sum(bool(x['blocking_gates'])for x in plans),'gates':dict(collections.Counter(g for x in plans for g in x['blocking_gates'])),'census_path':str(cp),'census_sha256':sha(cp),'protocol_path':str(protocol_path)if protocol_path.is_file()else None,'protocol_sha256':sha(protocol_path)if protocol_path.is_file()else None,'effective_config_tool_sha256':sha(module_path),'proposal_script_sha256':sha(__file__),'commands_executed':False,'profiled_roofline_rows':0,'remaining_ROI_requirement':'Even ready input plans require exact per-stage/kernel launch mapping; no automatic compute/memory classification.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in('gates','terminal_hardware_slots')},indent=2))
if __name__=='__main__':main()
