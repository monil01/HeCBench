#!/usr/bin/env python3
"""CPU-only fresh all4 miniDGS plans plus installed-NCU process-filter proposal.
Copies only byte-identical reviewed Python adapter; no compiler/GPU/MPI execution.
"""
import argparse,copy,hashlib,json,pathlib,subprocess,shutil
P=pathlib.Path;MODELS=('cuda','cuda_jl','ka','jacc')
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):P(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=P,default=P('/home/imo/HeCBench_julia'));ap.add_argument('--base-root',type=P,default=P('/tmp/hecbench-roofline-v100-20261020/batch023-completion-a'));ap.add_argument('--output-root',type=P,default=P('/tmp/hecbench-roofline-v100-20261020/batch040-miniDGS'));ap.add_argument('--packet',type=P,default=P(__file__).resolve().parent);a=ap.parse_args();root=a.source_root.resolve();out=a.output_root.absolute();packet=a.packet.resolve();ncu=P('/usr/lib/nsight-compute/ncu')
 if not out.resolve().is_relative_to(P('/tmp'))or out.resolve()==P('/tmp')or out.exists():raise ValueError('Fresh /tmp batch required')
 if shutil.disk_usage(out.parent).free<2*1024**3:raise ValueError('2GiB reserve required; no capture arrays staged now')
 packet.mkdir(parents=True,exist_ok=True)
 for flag,name in [('--help','installed_ncu_help.txt'),('--version','installed_ncu_version.txt')]:
  p=packet/name
  if p.exists():raise ValueError('Refuse overwriting CLI evidence')
  c=subprocess.run([str(ncu),flag],capture_output=True,text=True,check=True);p.write_text(c.stdout+c.stderr)
 helptext=(packet/'installed_ncu_help.txt').read_text()
 for term in ['--target-processes-filter','Only exactly matched','The executable name will be considered as process name']: 
  if term not in helptext:raise ValueError('Installed CLI lacks expected process-filter contract')
 selector={'benchmark':'miniDGS','configuration_id':'miniDGS_N6_full1022tet_2MPIranks_singleGPU6_5stageRK_textureobject_prealigned_v2','fixture_description':{'path':str(root/'src/miniDGS-julia/study_plan.json'),'sha256':sha(root/'src/miniDGS-julia/study_plan.json')},'selection':{m:['--filter-mode','per-launch-config','--kernel-name-base','mangled','--kernel-id','::regex:.*:1','--target-processes-filter','native_rank'if m=='cuda'else'julia']for m in MODELS},'expected_matching_launches':None,'coverage':'First mangled function/geometry per selected executable; excludes profiling launcher/utility executable names. Source order, rank PID identity and complete ROI remain unproved.','process_filter_scope':'Inclusion changes profiling targets, not MPI launch/rank count. Excluded-host injection avoidance and EBADF recovery are unproven.','expected_GPU_worker_processes':2,'MPI_ranks_preserved':2,'physical_GPUs':[6],'conditional_replay_review_preserved':True,'does_not_claim_replay_transport_validated':True}
 out.mkdir(parents=True);selectorpath=out/'process_filter_selector.json';dump(selectorpath,selector);summary=[];allbindings={str(ncu):sha(ncu),str(packet/'installed_ncu_help.txt'):sha(packet/'installed_ncu_help.txt'),str(packet/'installed_ncu_version.txt'):sha(packet/'installed_ncu_version.txt'),str(P(__file__).resolve()):sha(P(__file__).resolve()),str(selectorpath):sha(selectorpath)}
 for m in MODELS:
  bp=a.base_root/'miniDGS'/m/'plan.json';base=json.loads(bp.read_text());basehash=sha(bp);slot=out/'miniDGS'/m;scratch=slot/'scratch';slot.mkdir(parents=True);scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();(scratch/'profile_source').mkdir();old=base['scratch_cwd'];patches=base['completion_overlay_provenance']['patches']
  if len(patches)!=1 or P(patches[0]['original']).name!='study_driver.py':raise ValueError('Unexpected source overlay scope')
  patch=patches[0];text=patch['text'];oldcopy=patch['copy'];copysha=hashlib.sha256(text.encode()).hexdigest()
  if copysha!=patch['copy_sha256']or old in text:raise ValueError('Cannot reconstruct byte-identical path-independent driver')
  original=P(patch['original'])
  if sha(original)!=patch['original_sha256']:raise ValueError('Canonical Python driver changed')
  if "command=['mpiexec','-n','2',*rankcmd]"not in text or "for t in range(1,6):"not in text or "'native_rank'"not in text:raise ValueError('MPI2/full5/native basename proof missing')
  dest=scratch/'profile_source/study_driver.py';dest.write_text(text);dest.chmod(0o444)
  if sha(dest)!=copysha:raise ValueError('Staged driver byte mismatch')
  bindings={}
  for p,h in base['execution_bindings'].items():
   if p==oldcopy:bindings[str(dest)]=h
   else:
    if sha(p)!=h:raise ValueError('Canonical/helper/input binding drift '+p)
    bindings[p]=h
  replay=base['conditional_replay_review']
  if sha(replay['path'])!=replay['sha256']:raise ValueError('Conditional group-state proof changed')
  bindings.update(allbindings);bindings[str(bp)]=basehash;bindings[replay['path']]=replay['sha256'];plan=copy.deepcopy(base);plan['scratch_cwd']=str(scratch)
  for k in ['build_command','profile_target_command']:plan[k]=[x.replace(old,str(scratch))for x in plan[k]]
  plan['environment_overrides']={k:v.replace(old,str(scratch))for k,v in plan['environment_overrides'].items()}
  if plan['runtime_assets']or plan['runtime_asset_bytes']!=0:raise ValueError('Unexpected extra input staging')
  if plan['profile_target_command']!=['python3',str(dest),m]:raise ValueError('Unexpected target invocation')
  if plan['environment_overrides']['CUDA_VISIBLE_DEVICES']!='6':raise ValueError('Preserve declared GPU6')
  if base['blocking_gates']:raise ValueError('Do not clear any existing gate')
  rr=P(base['reference_result']);cc=rr.with_name('config.json')
  for p,h in [(rr,base['reference_result_sha256']),(cc,base['raw_config_sha256'])]:
   if sha(p)!=h:raise ValueError('Original numerical receipt changed')
   bindings[str(p)]=h
  plan['execution_bindings']=bindings;plan['staged']=True;plan['staging_status']='input_plan_ready_ROI_and_kernel_mapping_still_required';plan['process_filter_recovery_proposal']={'review_scope':'CPU source+installed CLI only; no GPU validation','base_plan':str(bp),'base_plan_sha256':basehash,'source_result_sha256':base['reference_result_sha256'],'source_config_sha256':base['raw_config_sha256'],'old_driver_copy_sha256':copysha,'new_driver_copy':str(dest),'new_driver_copy_sha256':sha(dest),'driver_bytes_identical_to_reviewed_023':True,'process_filter':selector['selection'][m][-1],'selector_path':str(selectorpath),'selector_sha256':sha(selectorpath),'conditional_replay_review':replay,'MPI_kernel_transport_or_check_changes':False,'kernel_group_checkpoint_requirements_unchanged':True,'original_safe_close_EBADF_preserved':True,'EBADF_fix_claim':False,'excluded_hosts_injection_absence_claim':False,'rank_PID_order_inference':False,'compiler_runs':0,'GPU_runs':0};plan['notes']+=['Fresh040 output paths and exact process-name profiling filter only. Two ranks/all five checks unchanged. Keep conditional group checkpoint/transport validity gates; no complete ROI claim. Use source-bound selection override externally; no kernel/MPI/transport changes.']
  proof=slot/'process_filter_source_proof.json';dump(proof,plan['process_filter_recovery_proposal']);bindings[str(proof)]=sha(proof);dump(slot/'plan.json',plan);dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in plan.items()if v!=base.get(k)});summary.append({'model':m,'plan':str(slot/'plan.json'),'plan_sha256':sha(slot/'plan.json'),'staged':True,'profile_process_filter':selector['selection'][m][-1],'GPU_runs':0,'build_runs':0})
  if sha(bp)!=basehash:raise ValueError('Base plan changed during staging')
 for p,h in allbindings.items():
  if sha(p)!=h:raise ValueError('Review input drift')
 dump(packet/'staging_summary.json',{'models':summary,'selector':str(selectorpath),'selector_sha256':sha(selectorpath),'staged_batch':str(out),'all4_source_identical_host_driver':True,'shared_source_or_transport_modifications':False,'compiler_runs':0,'GPU_runs':0,'next_run_requires_root_foreign_GPU_clear_and023_collector_completion':True,'postrun_required':['two GPU worker PIDs, no rank identity by PID ordering','both rank JSONs correct for each trial1..5','all five wrapper scientific checks/full artifacts/guards','source and selected config binding rechecks','retain original EBADF and any new failure; no profile success from merely matching zero kernels','keep counter/copy/kernel replay group restore and ROI partial qualifications']});print(json.dumps({'staged':len(summary),'batch':str(out),'selector':str(selectorpath),'no_GPU_or_build':True}))
if __name__=='__main__':main()
