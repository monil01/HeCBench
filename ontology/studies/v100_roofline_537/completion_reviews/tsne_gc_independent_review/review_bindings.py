"""Independent CPU-only source/metadata audit. No target imports or execution."""
import json,hashlib,re,difflib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def info(p):p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
plans=[]
for batch in ('batch051-tsne-native-python','batch052-gc-native-fixture','batch053-tsne-julia-python'):plans+=sorted((RAW/batch).glob('*/*/plan.json'))
cache={};records=[];sources={};findings=[]
for pp in plans:
 d=json.loads(pp.read_text());slot=pp.parent;checks=[];bad=[]
 for path,expected in d['execution_bindings'].items():
  if path not in cache:
   p=Path(path);cache[path]={'exists':p.is_file(),'actual_sha256':sha(p)if p.is_file()else None,'bytes':p.stat().st_size if p.is_file()else None}
  c=cache[path];checks.append({'path':path,'expected_sha256':expected,**c})
  if c['actual_sha256']!=expected:bad.append(path)
 result=Path(d['reference_result']);config=result.parent/'config.json';res=json.loads(result.read_text());cfg=json.loads(config.read_text())
 proofchecks=[]
 for p in slot.glob('*proof.json'):
  q=json.loads(p.read_text());sources[str(p)]=info(p)
  for patch in q.get('patches',[]):
   row={}
   for kind in ('source','copy'):
    path=patch.get(kind);expected=patch.get(kind+'_sha256')
    if path and expected:row[kind]={'path':path,'hash_matches':Path(path).is_file()and sha(path)==expected,'sha256':expected}
   if row:proofchecks.append(row)
 copycomparisons=[]
 for p in sorted((slot/'scratch/profile_source').glob('*')):
  if not p.is_file():continue
  sources[str(p)]=info(p)
  if d['benchmark']=='gc':original=ROOT/'src/gc-julia/study_cuda.cu'
  elif p.name=='run_metric.py':original=ROOT/'src/tsne-julia/run_metric.py'
  elif p.name in('study_native.cu','study_common.jl'):original=ROOT/'src/tsne-julia/guarded_I1'/p.name
  else:continue
  a=original.read_text();b=p.read_text();normalized=a
  if p.name=='run_metric.py':
   normalized=normalized.replace('from power2_codec import convert,digest','sys.path.insert(0,"'+str(ROOT/'src/tsne-julia')+'");from power2_codec import convert,digest').replace('P=pathlib.Path(__file__).resolve().parent','P=pathlib.Path("'+str(ROOT/'src/tsne-julia')+'")').replace('/tmp/hecbench-portable-infrastructure-private/tsne',str(slot/'scratch/profile_outputs/private_capture/tsne'))
  elif p.name=='study_native.cu':
   normalized=normalized.replace('../../tsne-cuda/common.h',str(ROOT/'src/tsne-cuda/common.h')).replace('../tsne-cuda/common.h',str(ROOT/'src/tsne-cuda/common.h')).replace('"observed_fit.inc"','"'+str(ROOT/'src/tsne-julia/guarded_I1/observed_fit.inc')+'"').replace('/tmp/tsne-native-capture',str(slot/'scratch/profile_outputs')).replace('/tmp/hecbench-portable-infrastructure-private/tsne',str(slot/'scratch/profile_outputs/private_capture/tsne'))
  elif p.name=='study_common.jl':normalized=normalized.replace('@__DIR__','"'+str(ROOT/'src/tsne-julia/guarded_I1')+'"').replace('/tmp/hecbench-portable-infrastructure-private/tsne',str(slot/'scratch/profile_outputs/private_capture/tsne'))
  else:
   normalized=normalized.replace('../gc-cuda/main.cu',str(ROOT/'src/gc-cuda/main.cu')).replace('std::filesystem::path(__FILE__).parent_path()','std::filesystem::path("'+str(ROOT/'src/gc-julia')+'")')
   for lit in re.findall(r'"([^"\n]+)"',a):
    if '.porting-state/portable_julia/raw_sidecars/gc'in lit:normalized=normalized.replace(lit,str(slot/'scratch/profile_outputs/gc_sidecars'))
  match=normalized==b;diff=''.join(difflib.unified_diff(normalized.splitlines(True),b.splitlines(True),fromfile='expected_path_only',tofile=str(p)))
  (HERE/(d['benchmark']+'_'+d['model']+'_'+p.name+'.diff')).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(original),tofile=str(p))))
  copycomparisons.append({'original':info(original),'copy':info(p),'matches_explicit_path_only_transform':match,'unexplained_diff':diff})
 entry={'plan':info(pp),'benchmark':d['benchmark'],'model':d['model'],'configuration_id':d['configuration_id'],'selected_result':info(result),'selected_config':info(config),'result_hash_matches':sha(result)==d['reference_result_sha256'],'config_hash_matches':sha(config)==d['raw_config_sha256'],'configuration_ids_match':res['configuration_id']==cfg['configuration_id']==d['configuration_id'],'execution_binding_count':len(checks),'binding_mismatches':bad,'binding_inventory_file':d['benchmark']+'_'+d['model']+'_binding_validation.json','proof_checks':proofchecks,'source_equivalence':copycomparisons,'profile_target_command':d['profile_target_command'],'build_command':d['build_command'],'environment_overrides':d['environment_overrides'],'effective_original_timing_region':d['effective_original_timing_region'],'blocking_gates':d['blocking_gates'],'staged':d['staged'],'runtime_asset_bytes':d['runtime_asset_bytes'],'assessment':'source_only_no_runtime_validation'}
 if d['benchmark']=='tsne':entry['audit_dependency_gap']={'bare_python3_remains':True,'audit_PYTHONPATH':'/tmp/pasc-julia-plot-tools','audit_package_bound_paths':sum(p.startswith('/tmp/pasc-julia-plot-tools')for p in d['execution_bindings']),'outer_python_numpy_bound_count':len(d['checker_interpreter_review']['prospective_bindings']),'Julia_setenv_replaces_environment':d['model']!='cuda','outer_cache_policy_does_not_cover_Julia_audit':d['model']!='cuda'}
 dump(HERE/entry['binding_inventory_file'],checks);records.append(entry)
extra=[ROOT/'src/tsne-julia/audit_native.py',ROOT/'src/tsne-julia/power2_codec.py',Path('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/share/julia/base/cmd.jl')]
for p in extra:sources[str(p)]=info(p)
report={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent source/plan/schema/inputrole review of exact queued051/052/053; not runtime or replay qualification.','records':records,'supporting_sources':sources,'findings':[{'id':'TSNE_inner_audit_dependency_and_cache_role','affected':['051/tsne/cuda','053/tsne/cuda_jl','053/tsne/ka','053/tsne/jacc'],'severity':'prospective_dependency_qualification_gap','proof':'Copied native uses system(PYTHONPATH=/tmp/pasc-julia-plot-tools python3 audit_native.py). Copied Julia common line89 uses setenv of barepython3 with only PYTHONPATH pair. Julia Base cmd.jl documented+implemented replacement environment; outer private cache/bytecode settings absent in audit subprocess. All4bind zero files under that audit NumPy tree. No actual loaded interpreter/NumPy observation asserted.','recommendation':'Future freshplans only: bind actual systemPython and entire audit PYTHONPATH file tree including existing .pyc/extensions/metadata; preserve inner checker code, describe Julia inherited-cache inputs/read-onlyLL policy accurately. No tolerance/math/timedregion/fivecheck change.'}],'preserved_semantics':['TSNE I1 argument, MNIST60000,K32,iteration/fullpipeline, all5 native/Juliacontroller checks, allocationredzones/independentrawaudits and full lossless roundtrip codec remain source unchanged except reviewed paths.','GC exact15,064B canonical CSR fixture source path restored; original native include/rep3/warmup/5property certificates/rawcolorsidecars/timedreset/convergence contract unchanged. Historical023rc11 unique cause not proved.'],'limitations':['No target import, build, execution, GPU query, test or shared/inflight-plan edit.','First-specialization kernel selection does not become retainedtimed/fullROI evidence through this review.','Historical Python/compiler/runtime identities not reconstructed; new bindings prospective only.']}
dump(HERE/'review.json',report)
print(json.dumps({'plans':len(records),'unique_bound_files':len(cache),'badbindings':[(r['benchmark'],r['model'],len(r['binding_mismatches']))for r in records],'path_transform_unexplained':[(r['benchmark'],r['model'],c['copy']['path'])for r in records for c in r['source_equivalence']if not c['matches_explicit_path_only_transform']]}))
