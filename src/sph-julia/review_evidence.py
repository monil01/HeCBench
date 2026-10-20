#!/usr/bin/env python3
"""Read-only final source/receipt review, no GPU execution."""
import hashlib,json,pathlib,sys
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'tools'));from verify_portable_julia import digest_tree
from audit_saved import audit
runs={'default':'20261020T042914619777Z_sph','ballistic':'20261020T042855878298Z_sph'};out={'status':'pass','configuration_receipts':{},'source_bindings':{},'source_trees':[],'selected_artifacts':0,'selected_artifact_bytes':0,'all_source_gpu_bodies_unchanged':True}
for case,rid in runs.items():
 path=R/'.porting-state/portable_julia/runs'/rid/'result.json';r=json.load(open(path));assert r['matched_performance_valid'] and all(r['variant_verified'].values()) and r['reference_agreement'];assert r['input_prefix_count'] in (83,56971)
 a=audit(path,case);ap=P/f'final_{case}_full_array_audit.json';ap.write_text(json.dumps(a,indent=2)+'\n');out['selected_artifacts']+=len(a['rows']);out['selected_artifact_bytes']+=sum(z['bytes']for z in a['rows']);out['configuration_receipts'][case]={'path':str(path.relative_to(R)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((path.parent/'config.json').read_bytes()).hexdigest(),'full_array_audit':str(ap.relative_to(R)),'full_array_audit_sha256':hashlib.sha256(ap.read_bytes()).hexdigest(),'native_reference_repaired':True,'native_behavior_known_scientific_defect':True,'exclude_untouched_native_aggregates':True}
 for model,m in r['models'].items():
  assert m['status']=='native_check_pass' and m['complete'] and m['timing_valid'] and m['provenance_valid'];assert len(m['trials'])==5;assert m['mean_seconds']==sum(t['time_seconds']for t in m['trials'][1:])/4
  for f,h in m['adapter_file_hashes'].items():assert hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h,f;out['source_bindings'][f]=h
  assert digest_tree(m['source_path'])==m['source_hash'],m['source_path'];out['source_trees'].append(dict(case=case,model=model,path=m['source_path'],sha256=m['source_hash']))
out['source_binding_count']=len(out['source_bindings']);(P/'final_source_binding_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k in ('status','source_binding_count','selected_artifacts','selected_artifact_bytes')}))
