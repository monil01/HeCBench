#!/usr/bin/env python3
"""Prepare only the reviewed native SVD3x3 raw-output environment overlay.

No source patch, compilation, target/GPU execution, or accepted evidence write.
Requires a fresh /tmp output directory and exact hash-bound environment approval.
"""
import argparse,copy,hashlib,json,pathlib,shutil,sys
sys.dont_write_bytecode=True
P=pathlib.Path
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def under(p,r):
 try:P(p).resolve().relative_to(P(r).resolve());return True
 except ValueError:return False
def check(d):
 for p,h in d.items():
  if not P(p).is_file()or sha(p)!=h:raise ValueError('Binding drift '+p)
def dump(p,d):P(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--source-root',type=P,default=P(__file__).resolve().parents[1]);a.add_argument('--review',type=P,required=True);a.add_argument('--review-sha256',required=True);a.add_argument('--output-root',type=P,required=True);a.add_argument('--stage',action='store_true');a.add_argument('--asset-cap-mib',type=int,default=64);a.add_argument('--reserve-mib',type=int,default=2048);z=a.parse_args();R=z.source_root.resolve();O=z.output_root.absolute();SELF=P(__file__).resolve();G=R/'tools/v100_profile_write_guard.py'
 if not under(O,'/tmp')or O.resolve()==P('/tmp')or O.exists()or O.is_symlink():a.error('Fresh owned output strictly under /tmp required')
 if min(z.reserve_mib,z.asset_cap_mib)<0:a.error('Nonnegative storage bounds required')
 if sha(z.review)!=z.review_sha256:raise ValueError('Review hash mismatch')
 review=json.loads(z.review.read_text())['models']['svd3x3/cuda'];f=P(review['source_plan_path'])
 if sha(f)!=review['source_plan_sha256']:raise ValueError('Approved base plan changed')
 base=json.loads(f.read_text());p=copy.deepcopy(base)
 if (base['benchmark']!='svd3x3'or base['model']!='cuda'or base['execution_bindings']!=review['execution_bindings']or base['reference_result_sha256']!=review['source_result_sha256']or base['raw_config_sha256']!=review['source_config_sha256']):raise ValueError('Exact review/plan join mismatch')
 rp=P(base['reference_result']);cp=rp.with_name('config.json')
 if sha(rp)!=base['reference_result_sha256']or sha(cp)!=base['raw_config_sha256']:raise ValueError('Accepted evidence drift')
 check(base['execution_bindings'])
 for q in review['source_evidence']:
  if sha(q['path'])!=q['sha256']:raise ValueError('Getter source changed')
 if review['channel']!='profile_target_command'or review['approved_only_for_environment_output_role']is not True:raise ValueError('Unexpected approval scope')
 S=O/'svd3x3/cuda/scratch';slot=S.parent;old=base['scratch_cwd'];move=lambda x:x.replace(old,str(S))if isinstance(x,str)else x
 for k in ['profile_target_command','build_command']:p[k]=[move(x)for x in p[k]]
 i=review['argv_index']
 if p['profile_target_command'][i]!=review['original_assignment']:raise ValueError('Exact old env assignment mismatch')
 if review['replacement_assignment']!='STUDY_RAW_DIR={scratch}/profile_raw':raise ValueError('Unexpected output replacement')
 p['profile_target_command'][i]=review['replacement_assignment'].replace('{scratch}',str(S));p['scratch_cwd']=str(S);p['environment_overrides']={k:move(v)for k,v in p['environment_overrides'].items()}
 for asset in p['runtime_assets']:asset['destination']=move(asset['destination'])
 # Preserve precisely declared scalar fixture/oracle files, copied readonly.
 if p['runtime_asset_bytes']>z.asset_cap_mib*1024**2:raise ValueError('Asset copy budget exceeded')
 for asset in p['runtime_assets']:
  if not P(asset['source']).is_file()or not under(asset['destination'],S):raise ValueError('Only declared file inputs beneath scratch reviewed')
  for q in asset['source_files']:
   if sha(q['path'])!=q['sha256']:raise ValueError('Declared input drift')
 bindings={**base['execution_bindings'],str(SELF):sha(SELF),str(G):sha(G),str(z.review.resolve()):z.review_sha256,str(f.resolve()):sha(f),str(cp):sha(cp)};before=dict(base['execution_bindings']);slot.mkdir(parents=True)
 if p['build_command']:
  compiler=list(p['build_command']);p['build_command']=[sys.executable,str(G),'--allow-dir',str(S),'--',*compiler];dump(slot/'compiler_manifest.json',{'compiler_command':compiler,'guarded_build_command':p['build_command'],'write_guard_sha256':sha(G),'compiler_runs':0,'GPU_runs':0});bindings[str(slot/'compiler_manifest.json')]=sha(slot/'compiler_manifest.json')
 p['execution_bindings']=bindings;p['staged']=False
 # No existing gate is waived by an env assignment: all unrelated gates remain.
 p['review']={'path':str(z.review.resolve()),'sha256':z.review_sha256,'entry':{'reason':'Exact native SVD3x3 STUDY_RAW_DIR output-only redirection. Kernel/input/build numeric flags and all other env unchanged.','approval':review,'additional_current_readonly_input_bindings':{q['path']:q['sha256']for q in review['source_evidence']},'cleared_gates':[],'profile_only':True,'kernel_math_changes':False}}
 p['profiler_requirements']['filesystem_write_guard_required']=True;p['native_env_overlay_provenance']={'source_before':before,'source_after':None,'original_assignment':review['original_assignment'],'new_assignment':p['profile_target_command'][i],'planner_sha256':sha(SELF),'compiler_runs':0,'GPU_runs':0,'source_patches':[]}
 if z.stage and not p['blocking_gates']:
  if shutil.disk_usage(O).free<p['runtime_asset_bytes']+z.reserve_mib*1024**2:raise ValueError('Storage reserve prevents staging')
  S.mkdir();(S/'profile_outputs').mkdir();(S/'ncu_tmp').mkdir();(S/'profile_raw').mkdir()
  for asset in p['runtime_assets']:
   src=P(asset['source']);dest=P(asset['destination']);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);dest.chmod(0o444)
   if sha(dest)!=sha(src):raise ValueError('Staged input mismatch')
   bindings[str(dest)]=sha(dest)
  check(bindings);p['native_env_overlay_provenance']['source_after']=dict(before);p['native_env_overlay_provenance']['source_bytes_unchanged']=True;p['staged']=True
 dump(slot/'plan_diff.json',{k:{'before':base.get(k),'after':v}for k,v in p.items()if base.get(k)!=v});dump(slot/'plan.json',p);dump(O/'summary.json',{'models':1,'ready':not p['blocking_gates'],'staged':p['staged'],'planner_sha256':sha(SELF),'plan':str(slot/'plan.json'),'compiler_runs':0,'GPU_runs':0});print(json.dumps({'ready':not p['blocking_gates'],'staged':p['staged'],'plan':str(slot/'plan.json')}))
if __name__=='__main__':main()
