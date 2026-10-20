#!/usr/bin/env python3
"""Read-only final source/receipt/full-array audit join; no GPU."""
import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def tree(p):
 h=hashlib.sha256()
 for f in sorted(p.rglob('*')):
  if f.is_file()and f.suffix in('.jl','.cu','.cuh','.h','.cpp','.toml'):h.update(str(f.relative_to(p)).encode());h.update(f.read_bytes())
 return h.hexdigest()
def main():
 runs={'pincat':'20261020T060153233158Z_tgvnn','rectangular':'20261020T060045696702Z_tgvnn'};bindings={};trees=[];results={};count=0;size=0;private_bytes=0;tracebytes=0
 for case,rid in runs.items():
  rp=R/'.porting-state/portable_julia/runs'/rid/'result.json';r=json.load(open(rp));assert r['matched_performance_valid']and r['reference_agreement']and not r['reference_disagreement']and all(r['variant_verified'].values());ap=P/'native_diagnostics'/f'jacc_{case}_full_array_audit.json';a=json.load(open(ap));assert a['status']=='pass'and a['result_sha256']==sha(rp);count+=a['artifact_count'];size+=a['artifact_bytes'];private_bytes+=sum(x['nonunique_SVD_raw']['bytes']for x in a['rows']);tracebytes+=sum(x['bytes']for x in a['actual_untimed_third_iteration_trace'])
  results[case]={'result':str(rp.relative_to(R)),'result_sha256':sha(rp),'configuration_sha256':sha(rp.parent/'config.json'),'audit':str(ap.relative_to(R)),'audit_sha256':sha(ap),'mean_seconds':{k:v['mean_seconds']for k,v in r['models'].items()}}
  for model,data in r['models'].items():
   assert data['provenance_valid']and data['timing_valid']and len(data['trials'])==5;sp=pathlib.Path(data['source_path']);assert tree(sp)==data['source_hash'];trees.append({'case':case,'model':model,'path':str(sp),'sha256':data['source_hash']})
   for name,digest in data['adapter_file_hashes'].items():
    f=pathlib.Path(name);assert f.exists()and sha(f)==digest,name
    if name in bindings:assert bindings[name]==digest
    bindings[name]=digest
   for row in data['trials']:
    f=pathlib.Path(row['output_artifact']['saved_path']);assert f.exists()and sha(f)==row['output_artifact']['sha256']
 out={'benchmark':'tgvnn','status':'pass','runs':results,'source_bindings':bindings,'source_trees':trees,'source_binding_count':len(bindings),'source_trees_checked':len(trees),'selected_primary_artifacts':count,'selected_primary_artifact_bytes':size,'selected_raw_nonunique_SVD_factors':40,'selected_raw_nonunique_SVD_factor_bytes':private_bytes,'actual_warmup_trace_fields':80,'actual_warmup_trace_bytes':tracebytes,'full_array_audit_scope':'All40trialartifacts exactactualinputprefix+perfieldindependentoracle+guards;80actualthird-iterationtraces;40rawSVDfactorsorthogonality/weightedVT/reconstructioninvariants. No phasecanonicalization or GPU rerun.','environment_binding_scope':'Pinnedproject/Manifest/prefs recorded separately in environment_snapshot.json; projectfiles not automaticallyboundinspecs. Do not imply automaticpertrialenvironment-filelock. Actualcommandsusefrozenprojects, no dependency/environmentmutations authorized/performed.','tool_sha256':sha(pathlib.Path(__file__))};(P/'final_source_binding_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['status','source_binding_count','source_trees_checked','selected_primary_artifacts','selected_primary_artifact_bytes','actual_warmup_trace_bytes','selected_raw_nonunique_SVD_factor_bytes']}))
if __name__=='__main__':main()
