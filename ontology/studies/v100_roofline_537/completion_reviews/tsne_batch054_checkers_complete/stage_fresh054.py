#!/usr/bin/env python3
"""CPU-only fresh path copies/bindings; no target imports/builds/tests/GPU."""
import argparse,datetime,hashlib,json,os,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;RAW=Path('/tmp/hecbench-roofline-v100-20261020');MODELS=('cuda','cuda_jl','ka','jacc');AUDIT=Path('/tmp/pasc-julia-plot-tools')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def meta(p):p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def verify(bindings):
 for p,h in bindings.items():
  if not Path(p).is_file()or sha(p)!=h:raise ValueError('Binding changed '+p)
def move(x,old,new):
 if isinstance(x,str):return x.replace(str(old),str(new))
 if isinstance(x,list):return [move(v,old,new)for v in x]
 if isinstance(x,dict):return {move(k,old,new):move(v,old,new)for k,v in x.items()}
 return x
def refresh(x):
 if isinstance(x,list):return [refresh(v)for v in x]
 if not isinstance(x,dict):return x
 z={k:refresh(v)for k,v in x.items()}
 for role in ('source','copy'):
  if isinstance(z.get(role),str)and role+'_sha256'in z and Path(z[role]).is_file():z[role+'_sha256']=sha(z[role])
 for k,v in list(z.items()):
  if isinstance(k,str)and k.startswith('/')and isinstance(v,str)and len(v)==64 and Path(k).is_file():z[k]=sha(k)
 return z
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-root',type=Path,required=True);args=ap.parse_args();out=args.output_root.resolve()
 if out.exists()or out==Path('/tmp')or not out.is_relative_to('/tmp'):raise ValueError('Fresh/tmp output only')
 auditfiles=sorted(p for p in AUDIT.rglob('*')if p.is_file())
 if any(p.is_symlink()for p in AUDIT.rglob('*')):raise ValueError('Unexpected auditdependency symlink needs explicit role')
 if not auditfiles:raise ValueError('Existing audit NumPy dependencytree missing')
 deps={str(p):sha(p)for p in auditfiles};interpreter_aliases={}
 for name in ('/usr/bin/python3','/usr/bin/python3.12','/bin/python3'):
  p=Path(name)
  if not p.is_file():raise ValueError('Expected systemPython path missing '+name)
  deps[str(p)]=sha(p);deps[str(p.resolve())]=sha(p.resolve());interpreter_aliases[name]=str(p.resolve())
 if set(interpreter_aliases.values())!={'/usr/bin/python3.12'}:raise ValueError('SystemPython alias differs needs review')
 stdlib=Path('/usr/lib/python3.12');stdlibfiles=sorted(p for p in stdlib.rglob('*')if p.is_file())
 for p in stdlibfiles:deps[str(p)]=sha(p)
 # Preserve exact existing Python subprocesses and their source/env semantics.
 # Native inherits outer cache env; Julia setenv REPLACES env with PYTHONPATH only.
 envreview={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'Source-identical checker dependency/input-role completion; no Python/NumPy/check code changes','audit_tree':str(AUDIT),'audit_tree_files':len(auditfiles),'audit_tree_bytes':sum(p.stat().st_size for p in auditfiles),'system_python_aliases':interpreter_aliases,'Python_resolution':'Both inherited current PATH and default/usr/bin:/bin candidates resolve sameSHA/usr/bin/python3.12; historical/runtime interpreter not observed. Barepython3 kept unchanged, launch environment must preserve this resolution.','observed_preparation_PATH':os.environ.get('PATH'),'current_which_python3':shutil.which('python3'),'stdlib_root':str(stdlib),'stdlib_files':len(stdlibfiles),'outer_codec_python':'/home/imo/noback/conda/bin/python3.9','outer_numpy_version':'1.22.3','inner_numpy_metadata':meta(AUDIT/'numpy-2.5.4.dist-info/METADATA'),'cache_policy':{'outer':'FreshprivatePYTHONPYCACHEPREFIX +PYTHONDONTWRITEBYTECODE1 prevents unbound shared PYC reuse/writes.','native_inner_audit':'Unchanged std::system audit inherits outer cache/bytecode settings; PYTHONPATH overridden to exactboundexisting audit tree.','Julia_inner_audit':'Unchanged setenv REPLACES environment with PYTHONPATH only, drops privateprefix/no-bytecode/TMPDIR. Existing source/PYC/cache files now allbound read-only inputs; Landlock denies writes outside freshscratch. Python may attempt sharedPYC writes and handle deniedwrites; no runtime compatibility claimed.','no_blanket_private_cache_claim':True},'prospective_bindings':deps,'historical_interpreter_identity':'Unknown; no retroactive strengthening','target_imports':0,'GPU_runs':0,'build_runs':0,'tests':0,'runtime_validated':False,'scope_limitations':['Actual imported Python/NumPy/DSOs remain future runtime observations, not established by hash inventory.','If actual barepython3 resolves another binary, fail bindings/review and prepare explicit host-env correction; no fallback authorized.','No source math/input/tolerance/fivecheck/timedregion/kernel-selector/replay-policy change.']}
 out.mkdir();dump(HERE/'checker_dependency_review.json',envreview);dump(HERE/'inner_checker_file_inventory.json',[meta(p)for p in auditfiles]);summary=[];rows=[]
 for model in MODELS:
  oldbatch='batch051-tsne-native-python'if model=='cuda'else 'batch053-tsne-julia-python';oldslot=RAW/oldbatch/'tsne'/model;oldplan=oldslot/'plan.json';b=json.loads(oldplan.read_text());verify(b['execution_bindings'])
  if b['blocking_gates']or not b['staged']:raise ValueError('Predecessor not ready')
  result=Path(b['reference_result']);cfg=result.parent/'config.json'
  if sha(result)!=b['reference_result_sha256']or sha(cfg)!=b['raw_config_sha256']:raise ValueError('Selected config/result changed')
  slot=out/'tsne'/model;scratch=slot/'scratch';src=scratch/'profile_source';src.mkdir(parents=True)
  for dirname in('profile_outputs','ncu_tmp','pythoncache'):(scratch/dirname).mkdir()
  d=move(b,oldslot,slot);mappings=[]
  for p in sorted((oldslot/'scratch/profile_source').glob('*')):
   if not p.is_file():continue
   new=src/p.name;oldbytes=p.read_bytes();newbytes=oldbytes.replace(str(oldslot).encode(),str(slot).encode());new.write_bytes(newbytes)
   if new.read_bytes().replace(str(slot).encode(),str(oldslot).encode())!=oldbytes:raise ValueError('Source path inverse mismatch')
   mappings.append({'old':meta(p),'new':meta(new),'inverse_path_exact':True,'only_replacement':{'from':str(oldslot),'to':str(slot)}})
  assets=[]
  for asset in d['runtime_assets']:
   for sourcefile in asset['source_files']:
    if sha(sourcefile['path'])!=sourcefile['sha256']:raise ValueError('Runtime asset source changed')
   dest=Path(asset['destination']);dest.relative_to(scratch);source=Path(asset['source'])
   if not source.is_file():raise ValueError('Unexpected directoryasset needs explicitreview')
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
   if sha(dest)!=sha(source):raise ValueError('Staged asset mismatch')
   assets.append({'source':meta(source),'destination':meta(dest)})
  # Rebuild path-only source proof BEFORE binding its immutable JSON. Nested
  # dictkeys and embedded diff/textpaths are rebased, all copied sourceSHAs updated.
  proof=refresh(move(json.loads((oldslot/'source_proof.json').read_text()),oldslot,slot));dump(slot/'source_proof.json',proof)
  d['remaining_completion_proof']=proof;d=refresh(d)
  bindings=d['execution_bindings'];bindings.update(deps)
  bindings[str(HERE/'checker_dependency_review.json')]=sha(HERE/'checker_dependency_review.json');bindings[str(HERE/'inner_checker_file_inventory.json')]=sha(HERE/'inner_checker_file_inventory.json');bindings[str(Path(__file__).resolve())]=sha(__file__)
  bindings[str(oldplan)]=sha(oldplan);bindings[str(oldslot/'source_proof.json')]=sha(oldslot/'source_proof.json')
  bindings[str(slot/'source_proof.json')]=sha(slot/'source_proof.json')
  # Copy only small review metadata; prior attempt files stay untouched.
  reviewpath=slot/'checker_dependency_review.json';dump(reviewpath,envreview);bindings[str(reviewpath)]=sha(reviewpath)
  d['checker_interpreter_review']['cache_policy']='Outer3.9 freshprivateprefix/no-bytecode. Innernative inherits; innerJulia setenv drops cacheenv, exactexisting audit source/PYC/extensions now boundreadonly and sharedwrites denied byLL. No blanketprivatecache claim.'
  d['inner_checker_dependency_review']={'path':str(reviewpath),'sha256':sha(reviewpath),'runtime_validated':False,'unchanged_checker_code':True,'prospective_binding_count':len(deps)}
  recovery={'schema_version':1,'purpose':'Fresh exact051/053 source/path-equivalent observer plans with complete inneraudit dependency/input-role hashes','predecessor_plan':meta(oldplan),'predecessor_source_proof':meta(oldslot/'source_proof.json'),'old_new_path_mapping':{'from':str(oldslot),'to':str(slot),'recursive_dict_keys_and_values':True},'source_copies':mappings,'runtime_assets':assets,'runtime_asset_bytes':d['runtime_asset_bytes'],'selected_result':meta(result),'selected_config':meta(cfg),'I1_and_all5checks_preserved':True,'kernel_math_changes':False,'checker_logic_or_tolerances_changed':False,'timed_region_changed':False,'source_role_only_dependency_completion':True,'runtime_validated':False,'GPU_runs':0,'build_runs':0,'tests':0}
  dump(slot/'recovery_source_proof.json',recovery);bindings[str(slot/'recovery_source_proof.json')]=sha(slot/'recovery_source_proof.json');verify(bindings)
  d['commands_executed']=[];dump(slot/'plan.json',d);summary.append({'benchmark':'tsne','model':model,'plan':str(slot/'plan.json'),'sha256':sha(slot/'plan.json')});rows.append(recovery)
 dump(out/'summary.json',summary);dump(HERE/'review.json',{'schema_version':1,'source_only':True,'output_root':str(out),'models':rows,'dependency_review':meta(HERE/'checker_dependency_review.json'),'no_shared_or_prior_plan_edits':True,'GPU_runs':0,'build_runs':0,'tests':0,'runtime_validated':False});print(json.dumps({'plans':summary,'inner_dependency_files':len(deps),'runtime_asset_bytes_total':sum(r['runtime_asset_bytes']for r in rows),'GPU_runs':0,'build_runs':0,'tests':0}))
if __name__=='__main__':main()
