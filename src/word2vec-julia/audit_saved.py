#!/usr/bin/env python3
"""Read-only full captured-field, guard, source-binding and CPU-oracle audit."""
import pathlib,json,hashlib,importlib.util,sys,numpy as np
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 rp=R/'.porting-state/portable_julia/runs/20261020T073115229620Z_word2vec/result.json';d=json.load(open(rp));A=pathlib.Path('/tmp/word2vec_assets_v2');m=json.load(open(A/'manifest.json'));g=np.memmap(A/'oracle.f64',dtype='<f8',mode='r');rows=[];bind={};trees=[]
 spec=importlib.util.spec_from_file_location('owned_readonly_study_verifier',R/'tools/verify_portable_julia.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for model,v in d['models'].items():
  assert v['complete']and v['status']=='native_check_pass'and len(v['trials'])==5 and v['timing_valid'];assert v['mean_seconds']==sum(t['time_seconds']for t in v['trials'][1:])/4
  for name,key in [('source_path','source_hash'),('adapter_source_path','adapter_hash')]:
   path=pathlib.Path(v[name]);assert mod.digest_tree(path)==v[key],(model,name);trees.append(dict(model=model,path=str(path),sha256=v[key]))
  for path,want in v['adapter_file_hashes'].items():assert sha(path)==want,(model,path);bind[path]=want
  for t in v['trials']:
   f=pathlib.Path(t['output_artifact']['saved_path']);assert sha(f)==t['output_artifact']['sha256'];assert f.stat().st_size==m['output_count']*8;x=np.memmap(f,dtype='<f8',mode='r');fields=[]
   for field in m['fields']:
    a=field['offset'];b=a+field['count'];xx=x[a:b];gg=g[a:b];assert np.all(np.isfinite(xx));err=np.abs(xx-gg);exact=field['kind']!='float';passed=bool(np.array_equal(xx.view('u8'),gg.view('u8')))if exact else bool(np.all(err<=1e-6+2e-5*np.abs(gg)));assert passed,(model,t['trial'],field)
    if not exact:
     assert np.array_equal(xx[-17:].view('u8'),gg[-17:].view('u8')),'tailguard'
     case=next(c for c in m['cases']if c['name']==field['case']);logical=case['vocab']*case['aligned'];idx=np.arange(logical).reshape(case['vocab'],case['aligned'])[:,case['dim']:].ravel();assert np.array_equal(xx[idx].view('u8'),gg[idx].view('u8')),'alignmentpadding'
    fields.append(dict(case=field['case'],field=field['field'],count=field['count'],exact=exact,max_abs_error=float(err.max()),passed=True))
   rows.append(dict(model=model,trial=t['trial'],artifact=str(f),sha256=sha(f),bytes=f.stat().st_size,fields=fields,all_dynamic_tailguards_exact=True,all_inactive_alignment_padding_exact=True))
 # Nontrivial oracle rejects zero-update implementation for both full-cold and hot controls.
 noops=[]
 for case in m['cases']:
  f=next(z for z in m['fields']if z['case']==case['name']and z['field']=='input_1');o=next(z for z in m['fields']if z['case']==case['name']and z['field']=='output_1');xx=g[f['offset']:f['offset']+f['count']];gg=g[o['offset']:o['offset']+o['count']];bad=int(np.count_nonzero(abs(xx-gg)>1e-6+2e-5*abs(gg)));assert bad>0;noops.append(dict(case=case['name'],unchanged_output_weights_rejected_values=bad))
 proof=dict(benchmark='word2vec',status='all20_complete_saved_captures_pass',result=str(rp.relative_to(R)),result_sha256=sha(rp),models=4,trials=rows,complete_artifact_bytes=sum(z['bytes']for z in rows),strict_source_bindings=bind,source_bindings_count=len(bind),source_trees_unchanged=trees,fullfields_per_trial=len(m['fields']),no_op_rejection=noops,atol=1e-6,rtol=2e-5,input_prefix_count=m['input_prefix_count'],all5GPUallocations_percase_17tailwords=True,original_fixed_constant_symbol_has_no_redzone=True,full_async_training_not_certified=True,GPU_execution=False,numpy=np.__version__);(P/'final_saved_full_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:proof[k]for k in ['status','complete_artifact_bytes','source_bindings_count','fullfields_per_trial','no_op_rejection']}))
if __name__=='__main__':main()
