#!/usr/bin/env python3
"""Independent full-array audit, rebuilding every tensor property from saved GPU states."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];RAW=ROOT/'.porting-state/porting_logs/svd3x3-julia-shared/raw'
result_path=Path(sys.argv[1]).resolve();r=json.loads(result_path.read_text());plan=json.loads((P/'study_plan.json').read_text());bounds=json.loads((P/'predeclared_contract.json').read_text())['predeclared_budgets'];bindings=[]
for m,v in r['models'].items():
 if not v.get('complete'):continue
 for f,h in v['adapter_file_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,(m,f)
 bindings.append({'model':m,'bindings':len(v['adapter_file_hashes'])})
rows=[];raw_values=0;raw_files=0;maxima={k:0. for k in bounds if not k.startswith('actual_')}
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for model,v in r['models'].items():
 if not v.get('complete'):continue
 label='native'if model=='cuda'else model
 pids=sorted({f.name.split('_trial')[0]for f in RAW.glob(label+'_pid*_trial5_case0_output_final.bin')});assert len(pids)==1,(model,pids)
 for trial in v['trials']:
  a=trial['output_artifact'];ap=Path(a['saved_path']);assert h(ap)==a['sha256'];values=np.memmap(ap,dtype='<f8',mode='r');assert len(values)==plan['artifact_count'];off=0;cases=[];file_hashes={}
  for c in plan['cases']:
   n=c['n'];k=c['index'];base=RAW/(pids[0]+f'_trial{trial["trial"]}_case{k}');arrays={}
   for key,ln in [('input_before',9*n+32),('output_initial',21*n+32),('input_after',9*n+32),('output_final',21*n+32)]:
    path=Path(str(base)+'_'+key+'.bin');q=np.memmap(path,dtype='<f4',mode='r');assert len(q)==ln;arrays[key]=q;file_hashes[str(path.relative_to(ROOT))]=h(path);raw_files+=1;raw_values+=ln;assert (q[:16]==np.float32(91.25)).all()and(q[-16:]==np.float32(91.25)).all()
   assert np.array_equal(values[off:off+4],[n,7,256,n//256+1]);off+=4
   for key in ['input_before','output_initial','input_after']:
    q=arrays[key];assert np.array_equal(values[off:off+len(q)],q.astype('f8'));off+=len(q)
   expect=np.fromfile(P/f'case{k}_input_f32.bin',dtype='<f4');assert np.array_equal(arrays['input_before'][16:-16],expect);assert np.array_equal(arrays['input_after'],arrays['input_before']);assert (arrays['output_initial'][16:-16]==np.float32(-777)).all()
   A=arrays['input_after'][16:-16].reshape(9,n).T.astype('f8').reshape(n,3,3);O=arrays['output_final'][16:-16].reshape(21,n).T.astype('f8');assert np.isfinite(A).all()and np.isfinite(O).all();U=O[:,:9].reshape(n,3,3);S=O[:,9:12];V=O[:,12:].reshape(n,3,3);denom=np.maximum(1,np.linalg.norm(A,axis=(1,2)));sigma=np.linalg.svd(A,compute_uv=False);ss=np.sort(abs(S),axis=1)[:,::-1];rec=(U*S[:,None,:])@V.swapaxes(1,2)
   e={'normalized_Frobenius_reconstruction_error':np.linalg.norm(rec-A,axis=(1,2))/denom,'normalized_sorted_absolute_singular_value_error':np.max(abs(ss-sigma),axis=1)/denom,'U_orthogonality_Frobenius_error':np.linalg.norm(U.swapaxes(1,2)@U-np.eye(3),axis=(1,2)),'V_orthogonality_Frobenius_error':np.linalg.norm(V.swapaxes(1,2)@V-np.eye(3),axis=(1,2)),'det_U_or_V_distance_from_plus_one':np.maximum(abs(np.linalg.det(U)-1),abs(np.linalg.det(V)-1))}
   em={key:float(x.max())for key,x in e.items()}
   for key,x in em.items():assert x<=bounds[key],(model,trial['trial'],k,key,x,bounds[key]);maxima[key]=max(maxima[key],x)
   cases.append((arrays,(ss/denom[:,None]).ravel(),em))
  assert off==plan['input_prefix_count']
  for c,(arrays,ss,em)in zip(plan['cases'],cases):
   n=c['n'];assert np.allclose(values[off:off+3*n],ss,atol=2e-15,rtol=2e-15);off+=3*n;assert (values[off:off+32]==91.25).all();off+=32;assert (values[off:off+6]==1).all();off+=6
  assert off==len(values);rows.append({'model':model,'trial':trial['trial'],'artifact_path':str(ap.relative_to(ROOT)),'artifact_sha256':h(ap),'raw_state_file_hashes':file_hashes,'all_full_factorizations_pass':True,'all_actual_inputs_initial_outputs_readonly_guards_exact':True,'case_maxima':[x[2]for x in cases]})
receipt={'result':str(result_path.relative_to(ROOT)),'result_sha256':h(result_path),'models':len(bindings),'trials':len(rows),'raw_files':raw_files,'raw_f32_values':raw_values,'full_matrix_factorizations':sum(c['n']for c in plan['cases'])*len(rows),'all_pass':True,'numpy_LAPACK_and_tensor_oracle_independent_of_C_checker_and_source_svd':True,'raw_U_S_V_identity_claim':False,'budgets_unchanged_from_before_native_first_probe':True,'source_binding_rehash':bindings,'maxima':maxima,'rows':rows}
output=P/('raw_state_audit_'+str(len(rows))+'trials.json');output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k not in ['rows','source_binding_rehash']}))
