from pathlib import Path
import json,gzip,hashlib,copy,fcntl,datetime,shutil
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';P=Path('/tmp/roofline-completion-20261021/roi_membership_source_batch6_f');O=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def canon(o):return sha(json.dumps(o,sort_keys=True,separators=(',',':')).encode())
def encoded(o):return (json.dumps(o,indent=2)+'\n').encode()
inputs={}
def read(p):
 p=Path(p);b=p.read_bytes();assert str(p)not in inputs or inputs[str(p)]==b;inputs[str(p)]=b;return b
def bind(p):return {'path':str(p),'sha256':sha(read(p))}
oldbytes=read(P/'proposed_stage_patches.json');doc=json.loads(oldbytes);new=copy.deepcopy(doc);snap=json.loads(gzip.decompress(read(P/'selected_review_records.json.gz')));oldrecords={(r['record']['benchmark'],r['record']['model']):r for r in snap};checks=[];parentchecks=[];capturechecks=[];sourcechecks=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 indexpath=S/'roi_reviews/INDEX.json';indexbytes=read(indexpath);index=json.loads(indexbytes);parents={}
 for name,expected in doc['parent_reviews'].items():
  path=Path(name);packed=read(path);raw=gzip.decompress(packed);obj=json.loads(raw);rows=[r for r in index['cohorts']if W/r['path']==path];assert len(rows)==1
  row=rows[0];assert row['compressed_sha256']==sha(packed)and row['raw_sha256']==sha(raw)
  parents[name]=obj;binding={'compressed_sha256':sha(packed),'uncompressed_sha256':sha(raw)}
  parentchecks.append({'path':name,'original_binding':expected,'current_binding':binding,'changed':expected!=binding,'exact_INDEX_row':row,'INDEX_sha256':sha(indexbytes)})
  new['parent_reviews'][name]=binding
  dst=O/'current_parent_snapshot'/path.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(packed)
 (O/'current_parent_snapshot/INDEX.json').write_bytes(indexbytes)
 for i,patch in enumerate(doc['patches']):
  key=(patch['record_key']['benchmark'],patch['record_key']['model']);old=oldrecords[key];assert old['parent']==patch['parent_review']['path']and old['index']==patch['record_index'];r=old['record'];assert canon(r)==patch['expected_original_record_canonical_sha256']
  parent=parents[patch['parent_review']['path']];idx=patch['record_index'];current=parent['records'][idx];assert current==r
  locations=[j for j,v in enumerate(parent['records'])if(v['benchmark'],v['model'])==key];assert locations==[idx]
  stagechecks=[]
  for operation in patch['stage_patches']:
   st=current['stages'][operation['stage_index']];assert st['stage_id']==operation['stage_id']and canon(st)==operation['expected_original_stage_canonical_sha256'];assert st==r['stages'][operation['stage_index']]
   proof=operation['updates']['ROI_membership_proof']
   for field in ['selected_result','selected_config','selected_profile_manifest','selected_counter_derivation','actual_profile_source_bindings']:
    e=proof[field];assert sha(read(e['path']))==e['sha256']
   stagechecks.append({'stage_index':operation['stage_index'],'stage_id':st['stage_id'],'unchanged_original_canonical_sha256':canon(st),'frozen_update_canonical_sha256':canon(operation['updates'])})
  b=new['parent_reviews'][patch['parent_review']['path']];new['patches'][i]['parent_review']={'path':patch['parent_review']['path'],**b}
  checks.append({'benchmark':key[0],'model':key[1],'parent':patch['parent_review']['path'],'record_index':idx,'unique_current_record_indices':locations,'entire_original_target_record_equal':True,'record_canonical_sha256':canon(r),'stages':stagechecks})
 for cap in doc['selected_capture_bindings']:
  for field in ['ledger','manifest','plan','derivation','provenance']:
   e=cap[field];assert sha(read(e['path']))==e['sha256']
  plan=json.loads(gzip.decompress(read(cap['plan']['path'])));ledger=json.loads(read(cap['ledger']['path']));contract=ledger['accepted_named_contract']
  for pk,ck in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]:assert plan[pk]==contract[ck]
  state=ledger['models'][cap['model']];assert state['profiling_status']=='profile_process_completed'and W/state['profile_evidence_manifest']==Path(cap['manifest']['path'])
  capturechecks.append({'benchmark':cap['benchmark'],'model':cap['model'],'all_original_capture_and_ledger_bytes_equal':True,'configuration_id':plan['configuration_id'],'reference_result_sha256':plan['reference_result_sha256'],'raw_config_sha256':plan['raw_config_sha256']})
 for e in doc['sources']:
  assert sha(read(e['path']))==e['sha256'];sourcechecks.append(e)
 # Only permitted parent-binding replacements are considered; scientific proof/updates/capture metadata stay exact.
 allowed=copy.deepcopy(new);allowed['parent_reviews']=doc['parent_reviews']
 for p,q in zip(allowed['patches'],doc['patches']):p['parent_review']=q['parent_review']
 assert allowed==doc
 for old,newpatch in zip(doc['patches'],new['patches']):assert encoded(old['stage_patches'])==encoded(newpatch['stage_patches'])
 assert new['selected_capture_bindings']==doc['selected_capture_bindings']and new['sources']==doc['sources']
 changed=any(p['changed']for p in parentchecks);candidate=encoded(new)if changed else oldbytes;(O/'proposed_stage_patches.json').write_bytes(candidate)
 for name in ['proofpacket.json.gz','selected_review_records.json.gz','companion_scientific_observations.json','summary.json','guide_bindings.json']:
  (O/name).write_bytes(read(P/name))
 for path,raw in inputs.items():assert Path(path).read_bytes()==raw,('input_changed',path)
 verification={'status':'current_parent_rebase_verified_no_changes_needed'if not changed else'parent_only_rebase_verified','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_packet':bind(P/'proposed_stage_patches.json'),'rebased_packet':{'path':str(O/'proposed_stage_patches.json'),'sha256':sha(candidate)},'candidate_byte_identical_to_original':candidate==oldbytes,'no_blind_hash_refresh':True,'parent_INDEX_joins':parentchecks,'record_checks':checks,'selected_capture_checks':capturechecks,'unchanged_source_bindings':sourcechecks,'record_count':len(checks),'stage_update_count':sum(len(x['stages'])for x in checks),'frozen_proofpacket_byte_identical':read(P/'proofpacket.json.gz')==(O/'proofpacket.json.gz').read_bytes(),'entire_input_snapshot_stable':True,'metadata_lock':'LOCK_SH whole current-parent/INDEX/target/config/capture/source read and before/after validation','shared_metadata_writes':False,'GPU_build_tests':0}
 (O/'rebase_verification.json').write_bytes(encoded(verification))
(O/'REVIEW.md').write_text('''# Private batch6 f v2 parent rebase

Original batch6 packet preserved. New v2 was constructed only after comparing every target record to the original compressed selected-record snapshot, checking original canonical record/stage hashes, exact unique record index, current parent gzip/raw hashes against its exact INDEX cohort row, every selected ledger/manifest/plan/provenance/derivation and config/result join, and all110 source bindings. All reads and before/after byte-stability checks ran under existing metadata LOCK_SH.

Observed result: batch6 has one parent, batch003_names161_240_review.json.gz; its current bytes and both parent hashes already equal the original packet. No parent refresh was necessary. All24 target records,103 stages,24 selected capture/ledger sets and110 source bindings remain unchanged. V2 proposed_stage_patches.json is byte-identical to the original; this is a verified no-op rebase, not evidence that any new scientific review or adoption occurred. The source proofpacket, original selected snapshot, companion observations, summary and guide bindings are copied byte-identically. Current parent/INDEX snapshot and full exact target proof are preserved separately in rebase_verification.json and current_parent_snapshot/.

Scientific updates, proofs, selectors, source hashes, capture joins and stage indices were not changed. Only parent binding replacement is permitted by the author, and no changed binding was observed. No phase/wholeROI/causal promotion, shared metadata mutation, Git, sudo, GPU, builds, tests or target imports/execution occurred. Registered usage capture is the sole authorized shared write. Root must still perform independent review and exact current-base checks at adoption.

A new prospective registered usage phase source_membership_batch6_f_rebase_20261021 was opened after startup/tool discovery and will be closed with raw exact events before final delivery. It is distinct from original batch6 source review and prior batch5 windows; postEND receipt packaging is excluded. Original source-review usage receipt remains at the original packet and is not relabeled or summed as rebase-only cost.
''')
manifest={'status':'PRIVATE_CURRENT_PARENT_REBASE_PENDING_ROOT_REVIEW','files':{str(p.relative_to(O)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}for p in sorted(O.rglob('*'))if p.is_file()and p.name!='DELIVERY_MANIFEST.json'}};(O/'DELIVERY_MANIFEST.json').write_bytes(encoded(manifest));print(json.dumps({k:verification[k]for k in ['status','candidate_byte_identical_to_original','record_count','stage_update_count','frozen_proofpacket_byte_identical','rebased_packet']},indent=2))
