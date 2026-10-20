from pathlib import Path
import json,hashlib,sys,datetime
R=Path('/home/imo/HeCBench_julia');Q=Path('/tmp/stencil-ports-537-outcome-audit/manifest_proposals_v2');review=json.loads((Q/'review.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=[]
for row in review['rows']:
 f=Q/f"{row['benchmark']}__{row['model']}.json"
 if not f.exists():continue
 proposal=json.loads(f.read_text());nr=proposal['normalization_review'];target=R/row['destination'];old=json.loads(target.read_text()) if target.exists() else {};prior=sha(target) if target.exists() else None
 # Independently rebind the reviewed source documents and the selected accepted receipt.
 for e in nr['source_evidence']:
  source=R/e['path'];assert sha(source)==e['sha256'],('Source evidence changed',source)
 rp=R/nr['selected_result'];assert sha(rp)==nr['selected_result_sha256'];d=json.loads(rp.read_text());assert d['reference_agreement'] and all(d['variant_verified'].values())
 suffix={'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}[row['model']];main=R/f"src/{row['benchmark']}-julia/{row['benchmark']}-julia-{suffix}/main.jl";assert sha(main)==nr['current_main_sha256'];assert d['models'][row['model']]['adapter_file_hashes'][str(main)]==sha(main)
 # Existing keys always win, including any concurrent coordinator source commit receipt.
 new=dict(proposal);new.update(old)
 # Fill reviewed aliases only where absent, preserving every original nested token object.
 for key,value in proposal.items():
  if key not in old or old[key] is None:new[key]=value
 if isinstance(old.get('translation_tokens'),dict) and isinstance(proposal.get('translation_tokens'),dict):
  nested=dict(proposal['translation_tokens']);nested.update(old['translation_tokens']);new['translation_tokens']=nested
 from_config=json.loads(rp.with_name('config.json').read_text())
 for key in ['native_reference_repaired','native_behavior_known_scientific_defect','verification_contract_revised','exclude_untouched_native_aggregates']:
  if key not in new:new[key]=bool(from_config.get(key,False))
 new['normalization_review']={**nr,'coordinator_applied_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_destination_sha256_at_proposal':row['original_sha256'],'destination_sha256_before_apply':prior,'current_fields_and_commit_receipts_preserved':True,'proposal_sha256':sha(f)}
 archive=R/f"src/{row['benchmark']}-julia/reference_history/accounting_original_delivery_20261020/{row['model']}_canonical_before_normalization_{prior[:12] if prior else 'missing'}.json"
 if old:
  archive.parent.mkdir(parents=True,exist_ok=True)
  if archive.exists():assert sha(archive)==prior
  else:archive.write_bytes(target.read_bytes())
 target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.reviewed.tmp');tmp.write_text(json.dumps(new,indent=2)+'\n');tmp.replace(target)
 out.append({'benchmark':row['benchmark'],'model':row['model'],'destination':row['destination'],'before_sha256':prior,'after_sha256':sha(target),'original_fields_preserved':all((all(new[k].get(j)==w for j,w in v.items()) if k=='translation_tokens' and isinstance(v,dict) else new[k]==v) for k,v in old.items() if k!='normalization_review')})
 assert out[-1]['original_fields_preserved']
p=R/'src/_portable_study/stencil_ports_537_audit_20261020/coordinator_manifest_normalization_receipt.json';p.write_text(json.dumps({'rows':out,'scope':'Coordinator source-bound add-only aliases, original actual counters and receipt fields retained. No measured sources/results/arrays/times changed.'},indent=2)+'\n');print('Applied',len(out),'reviewed additive manifests')
