# Own CPU-only JSON/text preparation. Does not import proposed tools or write /dev/shm.
from pathlib import Path
import hashlib,json,copy,os,difflib
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020')
P=Path('/tmp/roofline-completion-20261021/tmpfs_scratch_optin_proposal')
FUTURE=RAW/'batch063-su3-tmpfs';SHM=Path('/dev/shm/hecbench-v100-uid10384-su3_batch063_root_review_pending')
cache={}
def sha(p):
 p=Path(p);key=(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb')as f:
   for b in iter(lambda:f.read(4<<20),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def move(x,old,new):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,list):return [move(v,old,new) for v in x]
 if isinstance(x,dict):return {move(k,old,new):move(v,old,new) for k,v in x.items()}
 return x
limits=json.loads((P/'root_review_TEMPLATE_NOT_APPROVED.json').read_text())['limits']
review={'schema_version':1,'root_reviewed':False,'proposal_only':True,
 'base_collector_sha256':sha(P/'collector_C21_original.py'),
 'adopted_collector_sha256':sha(P/'proposed/tools/v100_roofline_collect.py'),
 'helper_sha256':sha(P/'proposed/tools/v100_tmpfs_scratch.py'),
 'guard_sha256':sha(R/'tools/v100_profile_write_guard.py'),
 'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',
 'limits':limits,'models':{},'held_models':{},
 'retention_review':'All fresh five-payload data retained in tmpfs; nondurable until separately archived with full bytes+SHA verification. No deletion authorized. Storage-only observer support, not replacement primary timing.',
 'execution_permission':False,'future_raw_plan_root':str(FUTURE),
 'follower_allowed_models':['cuda_jl','ka','jacc']}
rows=[]; inventory=[]; sources={};originals={}
for model in ('cuda','cuda_jl','ka','jacc'):
 original=RAW/('batch046-su3-native' if model=='cuda' else 'batch048-su3-julia')/'su3'/model/'plan.json'
 old=json.loads(original.read_text());scratch=SHM/model/'scratch';fresh=move(old,old['scratch_cwd'],str(scratch))
 assert move(fresh,str(scratch),old['scratch_cwd'])==old
 dest=D/'symbolic_plans'/'su3'/model/'plan.json';dump(dest,fresh)
 result=Path(old['reference_result']);config=result.with_name('config.json');res=json.loads(result.read_text());cfg=json.loads(config.read_text())
 assert sha(result)==old['reference_result_sha256'] and sha(config)==old['raw_config_sha256']
 trials=res['models'][model]['trials'];assert len(trials)==5
 trialjoins=[]
 for t in trials:
  a=t.get('output_artifact',{});trialjoins.append({'trial':t.get('trial'),'correct':t.get('correct'),
  'jit_excluded':t.get('jit_excluded'),'artifact':a,'time_seconds':t.get('time_seconds'),
  'internal_repeats':t.get('internal_repeats'),'timing_region':t.get('timing_region')})
 bindings=dict(fresh['execution_bindings']);bindings[str(result)]=sha(result);bindings[str(config)]=sha(config)
 for path,digest in old['execution_bindings'].items():
  assert sha(path)==digest,(path,'drift');sources[path]=digest
  if Path(path).is_relative_to(Path(old['scratch_cwd'])):
   relative=Path(path).relative_to(Path(old['scratch_cwd']));target=scratch/relative
   inventory.append({'model':model,'role':'held_native_source_copy' if model=='cuda' else 'scratch_readonly_source_copy',
    'source':path,'source_sha256':digest,'bytes':Path(path).stat().st_size,'future_destination':str(target),
    'copy_bytes_equal':True,'will_copy_by_follower':model!='cuda'})
 for p in (R/'tools/v100_profile_write_guard.py',R/'tools/v100_profiler_method.py',R/'tools/v100_roofline_classify.py',
           R/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json'):
  bindings[str(p)]=sha(p)
 entry={'original_plan_path':str(original),'original_plan_sha256':sha(original),
 'fresh_plan_sha256':sha(dest),'owned_root':str(SHM),'configuration_id':old['configuration_id'],
 'raw_config_path':str(config),'raw_config_sha256':sha(config),'reference_result_sha256':sha(result),
 'path_roles_independently_reviewed':model!='cuda',
 'storage_only_no_algorithm_ROI_or_checker_change':True,'read_only_bindings':bindings}
 if model=='cuda':
  entry['hold_reason']='Exact native source copy contains old046 absolute failure-only native_warm diagnostic prefix. Unchanged source bytes cannot redirect that branch by plan-prefix substitution; Landlock denies historical write. No native execution/approval in this packet.'
  review['held_models']['su3/'+model]=entry
 else:
  review['models']['su3/'+model]=entry
  copy_dest=D/'collector_plans'/'su3'/model/'plan.json';copy_dest.parent.mkdir(parents=True,exist_ok=True);copy_dest.write_bytes(dest.read_bytes())
 row={'benchmark':'su3','model':model,'original_plan':str(original),'original_plan_sha256':sha(original),
 'private_symbolic_plan':str(dest),'future_plan':str(FUTURE/'su3'/model/'plan.json'),'fresh_plan_sha256':sha(dest),
 'inverse_exact_entire_JSON_including_keys':True,'old_scratch_prefix':old['scratch_cwd'],'future_scratch_prefix':str(scratch),
 'reference_result':str(result),'reference_result_sha256':sha(result),'raw_config':str(config),'raw_config_sha256':sha(config),
 'five_accepted_primary_trial_metadata':trialjoins,'fresh_five_expected_outputs':[{'path':str(scratch/(('' if model=='cuda' else 'profile_outputs/')+'su3_trial'+str(i)+'.bin')),'bytes':1347849664} for i in range(1,6)],
 'source_readonly_input_bytes':673919456,'runtime_asset_copy_bytes':0,
 'new_primary_oracle_or_guard_semantics':False,'execution_eligible_after_root_review':model!='cuda'}
 rows.append(row); originals[str(original)]=sha(original)
 dump(D/'plan_diffs'/model/'paths.diff.json',{'old_scratch':old['scratch_cwd'],'new_scratch':str(scratch),
 'diff':''.join(difflib.unified_diff(json.dumps(old,indent=2).splitlines(True),json.dumps(fresh,indent=2).splitlines(True),fromfile=str(original),tofile=str(dest)))})
dump(D/'root_review_DRAFT_NOT_APPROVED.json',review)
dump(D/'all_four_plan_join_proofs.json',rows)
dump(D/'scratch_copy_inventory.json',{'files':inventory,'executing_Julia3_copy_bytes':sum(x['bytes'] for x in inventory if x['will_copy_by_follower']),
 'held_native_source_bytes':sum(x['bytes'] for x in inventory if not x['will_copy_by_follower']),
 'canonical_fixture_bytes_read_only':673919456,'canonical_fixture_copy_bytes':0,
 'compiler_helper_adoption':'Collector/helper tool adoption is separate root action, never copied automatically to source by follower.',
 'future_directories_each_eligible_model':['scratch','scratch/profile_outputs','scratch/ncu_tmp'],
 'dev_shm_paths_created':False})
dump(D/'before_bindings.json',{'original_plans':originals,'canonical_source_bindings':sources,
 'proposed_collector':{'path':str(P/'proposed/tools/v100_roofline_collect.py'),'sha256':sha(P/'proposed/tools/v100_roofline_collect.py')},
 'proposed_helper':{'path':str(P/'proposed/tools/v100_tmpfs_scratch.py'),'sha256':sha(P/'proposed/tools/v100_tmpfs_scratch.py')}})
print(json.dumps({'symbolic_plans':4,'eligible_after_root_review':3,'native_held':1,'runtime_asset_copies':0,'external_writes':0}))
