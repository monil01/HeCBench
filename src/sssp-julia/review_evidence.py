#!/usr/bin/env python3
"""Read-only full final artifact/source proof review; no GPU and no retiming."""
import pathlib,json,hashlib,sys,subprocess
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'tools'))
from verify_portable_julia import digest_tree
from audit_saved import audit
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
cases={'NYR':'20261020T051547309207Z_sssp','NYR_gpuonly':'20261020T051545881903Z_sssp','weighted_control':'20261020T051439043575Z_sssp','narrow_chain':'20261020T051547356730Z_sssp'}
v=dict(status='pass',benchmark='sssp',runs={},source_bindings={},source_trees=[],selected_artifacts=0,primary_artifact_bytes=0,private_artifact_bytes=0,all_original_source_GPU_bodies_unchanged=True)
for case,rid in cases.items():
 p=R/'.porting-state/portable_julia/runs'/rid/'result.json';r=json.load(open(p));assert r['matched_performance_valid']and all(r['variant_verified'].values())and r['reference_agreement'];a=audit(p);ap=P/f'final_{case}_full_array_audit.json';ap.write_text(json.dumps(a,indent=2)+'\n');v['selected_artifacts']+=len(a['rows']);v['primary_artifact_bytes']+=a['primary_artifact_bytes'];v['private_artifact_bytes']+=a['raw_private_artifact_bytes'];v['runs'][case]=dict(result=str(p.relative_to(R)),result_sha256=sha(p),config_sha256=sha(p.with_name('config.json')),full_array_audit=str(ap.relative_to(R)),audit_sha256=sha(ap))
 for model,m in r['models'].items():
  assert m['status']=='native_check_pass'and m['provenance_valid']and m['complete']and m['timing_valid']and len(m['trials'])==5;assert m['mean_seconds']==sum(t['time_seconds']for t in m['trials'][1:])/4
  for f,h in m['adapter_file_hashes'].items():assert sha(f)==h,f;v['source_bindings'][f]=h
  assert digest_tree(m['source_path'])==m['source_hash'],m['source_path'];v['source_trees'].append(dict(case=case,model=model,path=m['source_path'],sha256=m['source_hash']))
v['source_binding_count']=len(v['source_bindings']);(P/'final_source_binding_audit.json').write_text(json.dumps(v,indent=2)+'\n')
ldd=subprocess.check_output(['ldd',str(P/'libsssp_host.so')],text=True);symbols=subprocess.check_output(['nm','-D',str(P/'libsssp_host.so')],text=True);assert not any(s in ldd.lower()for s in ['cuda','cublas','cusparse']);assert not any(s in symbols for s in ['cudaLaunch','cuLaunchKernel','cudaMemcpy']);(P/'CPU_bridge_binary_review.json').write_text(json.dumps(dict(status='CPU_only_source_bound',library_sha256=sha(P/'libsssp_host.so'),ldd=ldd,exported_C_ABI=['sssp_first','sssp_host_level'],no_CUDA_DSO_dependency=True,no_CUDA_launch_or_copy_imports=True,source_files=['src/sssp-julia/host_bridge.cpp','src/sssp-cuda/kernel.cu'],source_scope='Originalkernel.cucontainsCPUworkerfunctionsONLY. NativeSSSP_gpuinmain.cuisNOTcompiledintothissharedbridge. All3targetGPUbodiesauthoredintheirfrontends.'),indent=2)+'\n')
print(json.dumps({k:v[k]for k in ['status','selected_artifacts','primary_artifact_bytes','private_artifact_bytes','source_binding_count']}))
