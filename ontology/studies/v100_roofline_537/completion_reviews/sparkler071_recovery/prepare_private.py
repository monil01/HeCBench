"""CPU-only source/recipe plan staging, no build/target/GPU/OS mutation."""
from pathlib import Path
import json,gzip,hashlib,fcntl,difflib,copy
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).parent.resolve();R=Path('/tmp/hecbench-roofline-v100-20261020/batch071-sparkler-privateMPI');HPC=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi');SRC=W/'src/sparkler-julia';UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d';sha=lambda b:hashlib.sha256(b).hexdigest();H=lambda p:sha(Path(p).read_bytes())
def B(p):return {'path':str(p),'sha256':H(p),'bytes':Path(p).stat().st_size}
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def tree(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [tree(v,a,b)for v in x]
 if isinstance(x,dict):return {k:tree(v,a,b)for k,v in x.items()}
 return x
assert not R.exists();R.mkdir();rows=[];archive=O/'retained_immutable';archive.mkdir();captured=[]
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);ledger=json.loads((S/'benchmarks/sparkler.json').read_bytes());contract=ledger['accepted_named_contract']
 for model in ['cuda','cuda_jl','ka','jacc']:
  manifest=W/ledger['models'][model]['latest_profile_attempt']['evidence_manifest'];parent=manifest.parent;planbytes=gzip.decompress((parent/'plan.json.gz').read_bytes());old=json.loads(planbytes);assert all(old[a]==contract[b]for a,b in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]);assert H(old['reference_result'])==old['reference_result_sha256']and H(Path(old['reference_result']).with_name('config.json'))==old['raw_config_sha256'];slot=R/'sparkler'/model;slot.mkdir(parents=True);scratch=slot/'scratch';scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();newsrc=scratch/'profile_source';newsrc.mkdir();ar=archive/model;ar.mkdir();(ar/'plan.json').write_bytes(planbytes);p=copy.deepcopy(old);patches=[];aliases=[]
  # Preserve every missing historical binding as an exact byte archival alias, never recreate oldpaths.
  recovered={x['copy']:x for x in old['completion_retry_proposal']['patches']}
  freshbindings={}
  for path,h in old['execution_bindings'].items():
   if Path(path).is_file():assert H(path)==h;freshbindings[path]=h;continue
   if path in recovered:
    data=recovered[path]['text'].encode();assert sha(data)==h
   elif Path(path).name=='compiler_manifest.json':
    actual=Path('/tmp/hecbench-roofline-v100-20261020/batch008-host-overlays/sparkler')/model/'compiler_manifest.json';assert H(actual)==h;data=actual.read_bytes()
   else:raise RuntimeError('Unrecoverable binding '+path)
   dest=ar/Path(path).name;dest.write_bytes(data);dest.chmod(0o444);freshbindings[str(dest)]=h;aliases.append({'historical_missing_path':path,'expected_sha256':h,'immutable_alias':B(dest),'recovery_basis':'Exact retained fulltext in pinned024plan'if path in recovered else 'Exact retained008manifest bytes; missing024path was erroneous pathrebase without copy','original_bytes_not_changed':True})
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:p[k]=tree(old[k],old['scratch_cwd'],str(scratch))
  assert p['runtime_asset_bytes']==0;p['execution_bindings']=freshbindings
  def writecopy(name,before,after,basis):
   assert before!=after;d=newsrc/name;d.write_text(after);d.chmod(0o444);patches.append({'copy':str(d),'copy_sha256':H(d),'before_sha256':sha(before.encode()),'before_text':before,'after_text':after,'basis':basis,'diff':''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='retained/'+name,tofile=str(d)))});p['execution_bindings'][str(d)]=H(d);return d
  for path,x in recovered.items():
   before=x['text'];assert sha(before.encode())==x['copy_sha256'];after=before.replace(old['scratch_cwd'],str(scratch))
   if Path(path).name=='study_launch.py':
    marker="cmd=[str(H/'ompi/bin/mpirun'),'--bind-to','none','-n','4']+sys.argv[1:]";assert after.count(marker)==1
    insert="mpi_base=Path(env['TMPDIR'])/'sparkler_mpi'\nmpi_base.mkdir(parents=True,exist_ok=False)\nsessions=mpi_base/'sessions';sessions.mkdir()\nbacking=mpi_base/'backing';backing.mkdir()\nvader=mpi_base/'vader';vader.mkdir()\nenv.update(OMPI_MCA_orte_tmpdir_base=str(sessions),OMPI_MCA_shmem_mmap_relocate_backing_file='1',OMPI_MCA_shmem_mmap_backing_file_base_dir=str(backing),OMPI_MCA_btl_vader_backing_directory=str(vader))\n"+marker
    after=after.replace(marker,insert);basis='Exact retained024launcher; append only installed MPI session/mmap/vader backing path controls inside fresh Landlock-authorized scratch; preserve rank4/ob1/self,vader,tcp/LD library roots.'
   else:basis='Exact retained024source; only fresh scratch path substitution.'
   dest=newsrc/Path(path).name
   if after==before:dest.write_text(after);dest.chmod(0o444);p['execution_bindings'][str(dest)]=H(dest);patches.append({'copy':str(dest),'copy_sha256':H(dest),'before_sha256':sha(before.encode()),'before_text':before,'after_text':after,'basis':'Exact byte-copy retained024source unchanged','diff':''})
   else:writecopy(dest.name,before,after,basis)
  # Latent numeric-only backend guard corrected outside source timer; headers anchored read-only.
  backend=SRC/'study_backend.cpp';before=backend.read_text();assert old['execution_bindings'].get(str(backend))==H(backend);a='study_need(visible&&std::string(visible)=="6","GPU6-only visibility required")';z='study_need(visible&&std::string(visible)=="'+UUID+'","exact GPU6 UUID visibility required")';assert before.count(a)==1;after=before.replace(a,z).replace('#include "study_capture.h"','#include '+json.dumps(str(SRC/'study_capture.h')));dest=writecopy('study_backend.cpp',before,after,'Only environment visibility guard after MPI_Init/before setup/timers and canonical read-only include anchor; no MPI/data/math/check/timer change.');assert after.replace(z,a).replace('#include '+json.dumps(str(SRC/'study_capture.h')),'#include "study_capture.h"')==before
  build=p['build_command'];assert build.count(str(backend))==1;build[build.index(str(backend))]=str(dest)
  # Old Julia build explicitly nests same unchanged Landlock guard. Keep it and absolute compiler.
  build=[('/usr/bin/nvcc'if x=='nvcc'else x)for x in build];p['build_command']=build;p['profile_target_command'][0]='/usr/bin/python3';p['environment_overrides']['CUDA_VISIBLE_DEVICES']='6'
  freshrecipe={'compiler_command':build,'compiler_executed':False,'GPU_executed':False,'historical_recipe_aliases':[x for x in aliases if Path(x['historical_missing_path']).name=='compiler_manifest.json'],'all_recipe_changes':'Same compiler/SDK/libraries/flags; fresh output/source paths, copied backend guard and absolute existing compiler only.'};save(slot/'fresh_compiler_manifest.json',freshrecipe);p['execution_bindings'][str(slot/'fresh_compiler_manifest.json')]=H(slot/'fresh_compiler_manifest.json')
  for f in [parent/'plan.json.gz',parent/'profile_provenance.json.gz',manifest,Path(__file__).resolve(),HPC/'lib/openmpi/mca_shmem_mmap.so',HPC/'lib/openmpi/mca_btl_vader.so',HPC/'lib/libopen-rte.so',HPC/'share/man/man1/mpirun.1',Path('/usr/bin/nvcc').resolve()]:p['execution_bindings'][str(f)]=H(f)
  assert all(Path(path).is_file()and H(path)==h for path,h in p['execution_bindings'].items());assert not p['blocking_gates']
  p['sparkler_private_MPI_recovery']={'parent_plan':B(parent/'plan.json.gz'),'parent_provenance':B(parent/'profile_provenance.json.gz'),'historical_missing_binding_aliases':aliases,'patches':patches,'MPI_ranks':4,'physical_GPU':'GPU6 exactUUID via unchangedcollector','transport':'ob1/self,vader,tcp unchanged','changed_path_roles':['orte session root','mmap sharedmemory backing','vader sharedmemory backing'],'global_dev_shm_or_guard_widening':False,'runtime_recovery_proven':False,'all_original_warmup_and_fivechecks_required':True,'scientific_or_timer_changes':False,'source_MPI_buffers':'Host stack reductions/gather/Bcast and barriers only, no device pointer MPI call; per-rank CUDAalloc/stream/vendorhandle/privatebuffers preserved. Kernelreplay safety remains conditional; no measured phase/fullROI claim.'}
  p['staged']=True;p['commands_executed']={'build':False,'GPU':False,'target':False};p['notes']+=['Fresh privately proposed sourcebound Sparkler MPIpath/immutablemanifest/UUIDguard recovery; rootreview required, runtimefix unproved.'];save(slot/'plan.json',p);save(slot/'original_plan.json',old);save(slot/'plan_diff.json',{k:{'before':old.get(k),'after':v}for k,v in p.items()if old.get(k)!=v});rows.append({'model':model,'plan':B(slot/'plan.json'),'source_recoveries':aliases,'patches':len(patches),'parent_provenance':B(parent/'profile_provenance.json.gz')})
 fcntl.flock(lock,fcntl.LOCK_UN)
save(O/'plans.json',{'rows':rows,'root':str(R),'rootreview_required':True,'GPU_build_target_runs':0});print(json.dumps({'plans':len(rows),'sourcealiases':sum(len(x['source_recoveries'])for x in rows),'root':str(R)}))
