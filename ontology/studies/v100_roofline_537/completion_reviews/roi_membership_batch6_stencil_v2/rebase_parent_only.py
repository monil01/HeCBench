from pathlib import Path
import hashlib,json,gzip,fcntl,copy,datetime
O=Path(__file__).resolve().parent;OLD=O.with_name('roi_membership_batch6_stencil');W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537'
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
p=OLD/'proposed_stage_patches.json';old=json.loads(p.read_bytes());new=copy.deepcopy(old);rows=[];parents={};checks=[]
with open('/tmp/hecbench-roofline-metadata.lock','r')as f:
 fcntl.flock(f,fcntl.LOCK_SH)
 for v in old['sources']:
  assert sha(Path(v['path']).read_bytes())==v['sha256'];checks.append({**v,'unchanged':True})
 for r in old['selected_capture_bindings']:
  for k in ['ledger','manifest','plan','derivation']:
   v=r[k];assert sha(Path(v['path']).read_bytes())==v['sha256'],('selected binding changed',r['benchmark'],r['model'],k)
   checks.append({**v,'unchanged':True})
 for pr in new['patches']:
  path=Path(pr['parent_review']['path']);b=path.read_bytes();raw=gzip.decompress(b);doc=json.loads(raw);key=pr['record_key'];idx=pr['record_index'];r=doc['records'][idx]
  assert all(r[k]==v for k,v in key.items()),('index/key mismatch',key,idx)
  assert canon(r)==pr['expected_original_record_canonical_sha256'],('target record changed',key)
  stages=[]
  for sp in pr['stage_patches']:
   st=r['stages'][sp['stage_index']];assert st['stage_id']==sp['stage_id'];assert canon(st)==sp['expected_original_stage_canonical_sha256'],('target stage changed',key,sp['stage_index'])
   stages.append({'stage_index':sp['stage_index'],'stage_id':sp['stage_id'],'canonical_sha256':canon(st),'updates_canonical_sha256':canon(sp['updates']),'proof_canonical_sha256':canon(sp['updates']['ROI_membership_proof'])})
  current={'compressed_sha256':sha(b),'uncompressed_sha256':sha(raw)};rows.append({'record_key':key,'record_index':idx,'exact_same_index':True,'record_canonical_sha256':canon(r),'parent_before':pr['parent_review'],'parent_after':{'path':str(path),**current},'stages':stages,'unchanged_updates_and_proofs':True})
  pr['parent_review']={'path':str(path),**current};parents[str(path)]=current
 new['parent_reviews']=parents
 # Only compressed/raw parent document references may differ. Every update and proof remains byte-canonical equal.
 for before,after in zip(old['patches'],new['patches']):
  assert canon(before['stage_patches'])==canon(after['stage_patches'])
  aa=copy.deepcopy(after);aa['parent_review']=before['parent_review'];assert canon(aa)==canon(before)
 fcntl.flock(f,fcntl.LOCK_UN)
(O/'proposed_stage_patches.json').write_text(json.dumps(new,indent=2)+'\n')
review={'schema_version':1,'scope':'Parent-document hash-only rebase; old packet retained; no source/proof/selected capture change','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_packet':{'path':str(p),'sha256':sha(p.read_bytes())},'rebased_packet':{'path':str(O/'proposed_stage_patches.json'),'sha256':sha((O/'proposed_stage_patches.json').read_bytes())},'exact_targeted_record_count':len(rows),'record_index_and_canonical_equality':True,'all_stage_update_and_proof_canonical_bytes_unchanged':True,'rows':rows,'selected_and_source_bindings':checks,'GPU_runs':0,'build_runs':0,'tests':0,'shared_edits':False}
(O/'rebase_proof.json').write_text(json.dumps(review,indent=2)+'\n')
(O/'README.md').write_text('Parent-only rebase of original batch6. Root must recheck current parent at merge. Exact targeted record/stage/index equality, all updates/proofs unchanged, and all original selected ledger/config/result/capture/source bindings unchanged. Original packet and receipt remain separate. See rebase_proof.json. No GPU/build/tests/shared modifications.\n')
print(json.dumps({'records':len(rows),'parents_changed':sum(x['parent_before']!=x['parent_after']for x in rows),'packet_sha256':review['rebased_packet']['sha256'],'proof_sha256':sha((O/'rebase_proof.json').read_bytes())}))
