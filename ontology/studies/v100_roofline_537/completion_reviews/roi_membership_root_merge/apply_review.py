"""Root-reviewed research metadata merge; validates actual evidence, no GPU/tests."""
from pathlib import Path
import collections, copy, datetime, fcntl, gzip, hashlib, json, re, shutil

ROOT=Path('/home/imo/HeCBench_julia')
STUDY=ROOT/'ontology/studies/v100_roofline_537'
WORK=Path('/tmp/roofline-completion-20261021')
OUT=WORK/'roi_membership_root_merge'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(obj):return sha(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())
def load(path):return json.loads(Path(path).read_bytes())
def check_binding(binding):
 p=Path(binding['path']);raw=p.read_bytes()
 expected=binding.get('sha256')or binding.get('compressed_sha256')
 assert sha(raw)==expected,('binding drift',str(p))
 if 'bytes'in binding:assert len(raw)==binding['bytes']
 uncompressed=binding.get('uncompressed_sha256')or binding.get('raw_sha256')
 if uncompressed:assert sha(gzip.decompress(raw))==uncompressed
 return json.loads(gzip.decompress(raw)if p.suffix=='.gz'else raw)

af=load(WORK/'roi_membership_AF/patch.json')
gp=load(WORK/'roi_membership_GP/proposed_stage_patches.json')
qz=load(WORK/'roi_membership_QZ/membership_patch.json')
assert sha((WORK/'roi_membership_AF/patch.json').read_bytes())=='402eadd98217bbed5985d0f4b7b95d6140dc16245a3f78bcf8ba472e1def4520'
assert sha((WORK/'roi_membership_GP/proposed_stage_patches.json').read_bytes())=='268ae04a9aac7704c6e9c1ef7d70b10fa0881631a60f0b20684ade979967d368'
assert sha((WORK/'roi_membership_QZ/membership_patch.json').read_bytes())=='c1279ab93abdf6ce6697313be166ec09f38165502b4400c656456cee9232cd65'
ops=[];contexts={};parents={}
def parent(info):
 path=info['path'];expected=(info['compressed_sha256'],info.get('uncompressed_sha256')or info['raw_sha256'])
 if path in parents:assert parents[path]['expected']==expected
 else:parents[path]={'expected':expected}
 return path
af_parent=parent(af['parent_review'])
for x in af['stage_patches']:
 key=(x['benchmark'],x['model']);contexts[key]={'style':'af','bindings':x['selected_profile_bindings'],'sources':x['set']['ROI_membership_proof']['sources']}
 ops.append(dict(parent=af_parent,key=key,stage=x['stage_id'],stage_sha=x['parent_stage_sha256'],updates=x['set']))
for x in af['non_authority_regex_alias_adjustments']:
 ops.append(dict(parent=af_parent,key=(x['benchmark'],x['model']),stage=x['stage_id'],old_regex=x['old_model_kernel_regex'],updates=x['set'],alias=True))
gp_captures={(x['benchmark'],x['model']):x for x in gp['selected_capture_bindings']}
for x in gp['patches']:
 key=(x['record_key']['benchmark'],x['record_key']['model']);sources=[]
 for y in x['stage_patches']:
  proof=y['updates']['ROI_membership_proof']
  for z in [proof['exact_stage_callsite']]+proof.get('timed_closure_boundaries',[])+proof.get('evidence',[]):
   sources.append({'path':z['path'],'source_sha256':z.get('source_sha256')or z['sha256']})
  ops.append(dict(parent=parent(x['parent_review']),key=key,stage=y['stage_id'],record_sha=x['expected_original_record_canonical_sha256'],stage_sha=y['expected_original_stage_canonical_sha256'],updates=y['updates']))
 contexts[key]={'style':'gp','bindings':gp_captures[key],'sources':sources}
for mapping,is_alias in [(qz['stage_updates'],False),(qz['duplicate_regex_alias_updates'],True)]:
 for x in mapping.values():
  key=(x['benchmark'],x['model']);e=qz['evidence']['/'.join(key)];contexts[key]={'style':'qz','bindings':e,'sources':e['sources']}
  ops.append(dict(parent=parent(x['parent_review']),key=key,stage=x['stage_id'],record_sha=x['parent_review']['record_sha256'],stage_sha=x['previous_stage_sha256'],updates=x['updates'],alias=is_alias))
assert len(contexts)==59
receipt={'schema_version':1,'activity':'Root-adopted source-family membership enrichment','sample_phase':'unknown; source family inclusion is not actual timed-sample proof','whole_elapsed_ROI_claim':False,'causal_claim':False,'changes':[],'selected_capture_checks':[],'GPU_build_tests':0}
with open('/tmp/hecbench-roofline-metadata.lock','a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for path,p in parents.items():
  packed=Path(path).read_bytes();raw=gzip.decompress(packed)
  assert(sha(packed),sha(raw))==p['expected'],('parent review drift',path)
  p.update(packed=packed,raw=raw,original=json.loads(raw),doc=json.loads(raw))
 for key,c in contexts.items():
  b,m=key;binding=c['bindings'];ledger=load(STUDY/'benchmarks'/(b+'.json'));state=ledger['models'][m]
  assert state['profiling_status']=='profile_process_completed',('selected status changed',key)
  if c['style']=='af':
   plan=check_binding(binding['plan']);prov=check_binding(binding['provenance']);derived=check_binding(binding['derived']);check_binding(binding['manifest']);check_binding(binding['frozen_config']);check_binding(binding['frozen_result']);folder=Path(binding['plan']['path']).parent
  elif c['style']=='gp':
   plan=check_binding(binding['plan']);derived=check_binding(binding['derivation']);check_binding(binding['manifest']);folder=Path(binding['plan']['path']).parent;prov=json.loads(gzip.decompress((folder/'profile_provenance.json.gz').read_bytes()))
  else:
   plan=check_binding(binding['plan']);prov=check_binding(binding['provenance']);derived=check_binding(binding['derived']);folder=Path(binding['plan']['path']).parent;assert sha((folder/'manifest.json').read_bytes())==binding['manifest_sha256']
  assert ROOT/state['kernel_observations']['path']==folder/'annotated.json.gz',('selected capture drift',key)
  contract=ledger['accepted_named_contract'];assert plan['reference_result_sha256']==contract['result_sha256']and plan['raw_config_sha256']==contract['raw_config_sha256']and plan['configuration_id']==contract['configuration_id']
  assert prov['plan_sha256']==sha(gzip.decompress((folder/'plan.json.gz').read_bytes()))
  captured=collections.defaultdict(set)
  for mapping in [plan['execution_bindings'],prov.get('additional_current_profile_input_bindings',{})]:
   for path,digest in mapping.items():captured[str(Path(path).resolve())].add(digest)
  for source in c['sources']:
   path=Path(source['path']);digest=source.get('source_sha256')or source.get('sha256');assert sha(path.read_bytes())==digest,('source drift',str(path));assert digest in captured[str(path.resolve())],('source absent from actual profile bindings',key,str(path))
   if 'exact_lines'in source:
    lines=path.read_text().splitlines();assert '\n'.join(lines[source['start_line']-1:source['end_line']])==source['exact_lines'],('excerpt drift',key,str(path))
  c.update(plan=plan,derived=derived)
  receipt['selected_capture_checks'].append({'benchmark':b,'model':m,'profile':str(folder.relative_to(ROOT)),'source_binding_checks':len(c['sources']),'result_sha256':plan['reference_result_sha256'],'configuration_id':plan['configuration_id'],'current_ledger_sha256':sha((STUDY/'benchmarks'/(b+'.json')).read_bytes()),'ledger_scope':'Current selected contract/capture checked under lock; previous whole-ledger hash can differ through periodic metadata refresh.'})
 staged=set()
 for op in ops:
  identity=(*op['key'],op['stage']);assert identity not in staged,('duplicate patch',identity);staged.add(identity)
  p=parents[op['parent']];old=next(x for x in p['original']['records']if(x['benchmark'],x['model'])==op['key']);new=next(x for x in p['doc']['records']if(x['benchmark'],x['model'])==op['key'])
  if op.get('record_sha'):assert canonical(old)==op['record_sha'],('record drift',identity)
  stage=next(x for x in old['stages']if x['stage_id']==op['stage']);target=next(x for x in new['stages']if x['stage_id']==op['stage'])
  if op.get('stage_sha'):assert canonical(stage)==op['stage_sha'],('stage drift',identity)
  if 'old_regex'in op:assert stage.get('model_kernel_regex')==op['old_regex']
  updates=op['updates']
  if op.get('alias'):assert updates['model_kernel_regex']is None and 'semantic_family_in_primary_ROI'not in updates
  else:
   assert updates['ROI_membership_status']in['source_timed_closure_inclusion_proved','source_setup_exclusion_proved','source_untimed_audit_exclusion_proved']and updates['ROI_membership_proof']
   assert updates['semantic_family_in_primary_ROI']==(updates['ROI_membership_status']=='source_timed_closure_inclusion_proved')
  target.update(copy.deepcopy(updates));receipt['changes'].append({'benchmark':op['key'][0],'model':op['key'][1],'stage_id':op['stage'],'alias_only':bool(op.get('alias')),'membership':updates.get('semantic_family_in_primary_ROI'),'parent_stage_sha256':canonical(stage),'adopted_stage_sha256':canonical(target)})
 for key,c in contexts.items():
  record=next(x for p in parents.values()for x in p['doc']['records']if(x['benchmark'],x['model'])==key)
  proofmatches=0
  for obs in c['derived']['records']:
   matched=[x for x in record['stages']if x.get('model_kernel_regex')and re.search(x['model_kernel_regex'],obs['identity']['Kernel Name'])]
   assert len(matched)<=1,('ambiguous family identity',key,obs['identity']['Kernel Name'])
   proofmatches+=sum(bool(x.get('ROI_membership_proof'))and x.get('semantic_family_in_primary_ROI')is True for x in matched)
  assert proofmatches>0,('no promoted family matches saved capture',key)
 for path,p in parents.items():
  for old,new in zip(p['original']['records'],p['doc']['records']):
   old_other={k:v for k,v in old.items()if k!='stages'};new_other={k:v for k,v in new.items()if k!='stages'};assert old_other==new_other
  raw=(json.dumps(p['doc'],indent=2,allow_nan=False)+'\n').encode();packed=gzip.compress(raw,mtime=0);p.update(new_raw=raw,new_packed=packed)
 history=STUDY/'roi_review_history';history.mkdir(exist_ok=True)
 for path,p in parents.items():
  archived=history/(p['expected'][0]+'.json.gz')
  if archived.exists():assert archived.read_bytes()==p['packed']
  else:archived.write_bytes(p['packed'])
  Path(path).write_bytes(p['new_packed'])
 receipt['parents']=[{'path':path,'previous_compressed_sha256':p['expected'][0],'previous_raw_sha256':p['expected'][1],'archived':str((history/(p['expected'][0]+'.json.gz')).relative_to(ROOT)),'new_compressed_sha256':sha(p['new_packed']),'new_raw_sha256':sha(p['new_raw'])}for path,p in parents.items()]
 indexpath=STUDY/'roi_reviews/INDEX.json';oldindex=indexpath.read_bytes();(history/(sha(oldindex)+'.index.json')).write_bytes(oldindex);index=json.loads(oldindex)
 for row in index['cohorts']:
  path=str(ROOT/row['path'])
  if path in parents:row.update(raw_sha256=sha(parents[path]['new_raw']),compressed_sha256=sha(parents[path]['new_packed']))
 index['review_scope']='Source callsite candidates, with explicit source-bound closure-membership proofs for a reviewed subset; actual sampled invocation phases and complete elapsed ROI remain unproved.'
 index['enrichment_receipt']='ontology/studies/v100_roofline_537/completion_reviews/roi_membership_root_merge/root_adoption.json';indexpath.write_text(json.dumps(index,indent=2)+'\n')
 receipt.update(record_count=len(contexts),benchmark_count=len({k[0]for k in contexts}),inclusion_stages=sum(x['membership']is True for x in receipt['changes']),exclusion_stages=sum(x['membership']is False for x in receipt['changes']),matching_only_alias_adjustments=sum(x['alias_only']for x in receipt['changes']),time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='source_family_reviews_adopted_pending_fresh_annotation_integrity_audit')
 (OUT/'root_adoption.json').write_text(json.dumps(receipt,indent=2)+'\n')
 target=STUDY/'completion_reviews/roi_membership_root_merge';target.mkdir(exist_ok=True)
 shutil.copyfile(OUT/'root_adoption.json',target/'root_adoption.json');shutil.copyfile(Path(__file__),target/'apply_review.py')
print(json.dumps({k:v for k,v in receipt.items()if k not in['changes','selected_capture_checks','parents']}))
