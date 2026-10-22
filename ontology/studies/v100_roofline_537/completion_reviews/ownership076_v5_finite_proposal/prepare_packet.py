from pathlib import Path
import json,gzip,hashlib,fcntl,shutil,ast
P=Path(__file__).parent;W=Path('/home/imo/HeCBench_julia');R=Path('/tmp/hecbench-roofline-v100-20261020/batch076-ownership-calibration');A=P/'future_runtime_authority'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 ledger=W/'ontology/studies/v100_roofline_537/benchmarks/triad.json';ledgerraw=ledger.read_bytes();l=json.loads(ledgerraw)
 c=l['accepted_named_contract'];assert c['configuration_id']=='triad_kernel_only_n4194304_r100_shmem0_v2'
 oldpath=W/'ontology/studies/v100_roofline_537/profiles/triad/cuda/batch009-triad-retained/plan.json.gz';oldraw=gzip.decompress(oldpath.read_bytes());old=json.loads(oldraw)
 assert c['result_sha256']==old['reference_result_sha256'] and c['raw_config_sha256']==old['raw_config_sha256']
 assert str(W/c['result'])==old['reference_result']
plan=json.loads(oldraw);before=old['scratch_cwd'];after=str(R/'triad/cuda/scratch')
def rebase(x):
 if isinstance(x,str):return x.replace(before,after)
 if isinstance(x,list):return [rebase(y) for y in x]
 if isinstance(x,dict):return {k:rebase(v) for k,v in x.items()}
 return x
plan=rebase(plan);assert rebase(old)==plan
# Exact inverse, including all unmodified scientific fields.
def inverse(x):
 if isinstance(x,str):return x.replace(after,before)
 if isinstance(x,list):return [inverse(y) for y in x]
 if isinstance(x,dict):return {k:inverse(v) for k,v in x.items()}
 return x
assert inverse(plan)==old
(P/'accepted009_plan_original.json').write_bytes(oldraw)
dump(P/'plan076.json',plan)
dump(P/'plan_inverse_proof.json',{'original_path':str(oldpath),'original_gzip_sha256':sha(oldpath),'original_decompressed_sha256':hashlib.sha256(oldraw).hexdigest(),'new_plan_sha256':sha(P/'plan076.json'),'sole_substitution':{'before':before,'after':after},'exact_full_object_inverse_equal':True,'original_full_text_retained':'accepted009_plan_original.json','math_checks_timers_build_inputs_selectors_metrics_unchanged':True})
T=P/'authority_template'
for rel,src in [('tools/v100_roofline_collect_ownership_proposal.py',P.parent/'own_teardown_guard_proposal_v5_1/v100_roofline_collect_ownership_proposal.py'),('tools/v100_profiler_method.py',W/'tools/v100_profiler_method.py'),('tools/v100_profile_write_guard.py',W/'tools/v100_profile_write_guard.py'),('tools/v100_roofline_classify.py',W/'tools/v100_roofline_classify.py'),('ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json',W/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json')]:
 dest=T/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
privateledger=T/'ontology/studies/v100_roofline_537/benchmarks/triad.json';privateledger.parent.mkdir(parents=True,exist_ok=True);privateledger.write_bytes(ledgerraw)
selector=W/'ontology/studies/v100_roofline_537/ROI_triad_retained_trial2.json';s=json.loads(selector.read_bytes());assert s['configuration_id']==plan['configuration_id'] and s['expected_matching_launches']==100
(P/'selector009_exact.json').write_bytes(selector.read_bytes());row=next(x for x in s['source_records'] if x['model']=='cuda')
assert row['reference_result']['sha256']==plan['reference_result_sha256'] and row['raw_config']['sha256']==plan['raw_config_sha256']
assert row['source_ROI']['timed_device_launches_per_trial']==100 and row['launch_ledger']=={'warmup':1,'per_trial':100,'trial2_start':102,'total_scientific':501}
assert s['selection']['cuda']==row['selectors'][0]['selector']
bindings={}
def bind(p,expected=None):
 p=Path(p);h=sha(p);assert expected is None or h==expected,(str(p),h,expected);bindings[str(p)]=h
# Avoid function/dictionary name collision.
for p,h in plan['execution_bindings'].items():bind(p,h)
bind(plan['reference_result'],plan['reference_result_sha256']);bind(row['raw_config']['path'],plan['raw_config_sha256']);bind(ledger);assert bindings[str(ledger)]==hashlib.sha256(ledgerraw).hexdigest()
bind(selector);bind(s['source_recipe']['path'],s['source_recipe']['sha256']);bind(s['fixture_description']['path'],s['fixture_description']['sha256']);bind(oldpath)
for r in row['execution_bindings']+row.get('current_supplemental_sources',[]):bind(r['path'],r['sha256'])
# Pin supplemental canonical source bytes; templates do not duplicate source/data trees.
source_dirs=[W/'src/triad-julia',W/'src/triad-cuda'];ext={'.jl','.py','.cu','.cuh','.c','.cpp','.h','.hpp','.inc','.toml'}
for directory in source_dirs:
 for p in sorted(directory.rglob('*')):
  if p.is_file() and p.suffix in ext:bind(p)
method=W/'ontology/studies/v100_roofline_537/profiler_methods/ncu2025_2_0_V100_sm70_raw_v1/review.json';mr=json.loads(method.read_bytes());bind(method,'65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca');bind(method.parent/mr['bundle_file'],mr['bundle_sha256'])
# Read frozen archive as JSON, never import or execute its source.
import base64
bundle=json.loads((method.parent/mr['bundle_file']).read_bytes());man=json.loads(base64.b64decode(bundle['files']['manifest.json']['base64']))
for name,h in man['bindings'].items():
 if name==mr['profiler']['executable'] or name.startswith(str(Path(mr['profiler']['executable']).parent)+'/'):bind(name,h)
for p in ['/usr/bin/python3','/usr/bin/nvcc','/usr/lib/nvidia-cuda-toolkit/bin/nvcc','/home/imo/HeCBench_julia/tools/v100_roofline_collect.py'] :bind(p)
for rel in ['tools/v100_profiler_method.py','tools/v100_profile_write_guard.py','tools/v100_roofline_classify.py','ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json']:bind(W/rel)
bind(P.parent/'own_teardown_guard_proposal_v5_1/v100_roofline_collect_ownership_proposal.py')
bind(P.parent/'ownership_guard_v5_1_independent_review/DELIVERY.json','d9358d514a6ded9a3840d9571c0153ca7b55cbefbd9b8f83f90f43ab19db67e4')
for p in T.rglob('*'):
 if p.is_file():bind(p)
for p in [P/'selector009_exact.json',P/'plan076.json',P/'accepted009_plan_original.json',P/'plan_inverse_proof.json',P.parent/'own_teardown_guard_proposal_v5_1/DELIVERY.json',P.parent/'own_teardown_guard_proposal_v5_1/source_inverse_proof.json',P.parent/'own_teardown_guard_proposal_v5/source_inverse_proof.json',P.parent/'own_teardown_guard_proposal_v4/source_inverse_proof.json']:bind(p)
# Keep exact003 authority references distinct from current runtime files.
raw003=Path('/tmp/hecbench-roofline-v100-20261020/method_calibration_ncu2025_2_0_003')
for name in ['profile_provenance.json','controller_source.py','collector_source.py','parser_source.py']:bind(raw003/name)
metric_source=(T/'tools/v100_roofline_collect_ownership_proposal.py').read_text();tree=ast.parse(metric_source)
metric_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='METRICS' for z in n.targets))
metric_expr=ast.get_source_segment(metric_source,metric_node.value)
argv=['/usr/bin/python3',str(A/'tools/v100_roofline_collect_ownership_proposal.py'),'--source-root',str(A),'--plan-root',str(R),'--benchmark','triad','--timeout','600','--reserve-mib','2048','--selection-overrides',str(P/'selector009_exact.json'),'--profiler-executable',mr['profiler']['executable'],'--profiler-method-review',str(method),'--profiler-method-review-sha256',bindings[str(method)]]
dump(P/'launch_spec.json',{'raw_root':str(R),'runtime_authority':str(A),'template':str(T),'source_root':str(W),'ledger_relative':'ontology/studies/v100_roofline_537/benchmarks/triad.json','canonical_ledger':str(ledger),'canonical_ledger_sha256':bindings[str(ledger)],'source_symlinks':{str(A/'src'/d.name):str(d) for d in source_dirs},'argv':argv,'environment':{'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'','LD_PRELOAD':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'},'candidate_sha256':sha(T/'tools/v100_roofline_collect_ownership_proposal.py'),'method_review_sha256':bindings[str(method)],'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d','max_collector_invocations':1,'requested_process_timeout_seconds':600,'reserve_bytes':2*1024**3,'minimum_initial_free_bytes':3*1024**3,'fresh_payload_bytes':5*4194304*4,'bindings':bindings,'metrics_source_expression':metric_expr,'method_id':mr['method_id'],'raw003_historical_method_only_not_current83claim':True})
dump(P/'selector_source_proof.json',{'selector_exact_copy_sha256':sha(P/'selector009_exact.json'),'accepted009_selector_sha256':sha(selector),'native_source_record':row,'source_counting':'one warmup + trial1 x100 => skip101; trial2 launches102..201 x100; --kill0 preserves subsequent full checks','source_ordinal_not_dynamic_trace':True,'no_full_native_transfer_pipeline_or_wholeROI_promotion':True})
print(json.dumps({'bindings':len(bindings),'candidate_sha256':sha(T/'tools/v100_roofline_collect_ownership_proposal.py'),'plan_sha256':sha(P/'plan076.json'),'runtime_staged':A.exists(),'raw_root_staged':R.exists()}))
