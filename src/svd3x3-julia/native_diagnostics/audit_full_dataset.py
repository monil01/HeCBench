#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;N=1048576;c=json.loads((P.parent/'predeclared_contract.json').read_text());bounds=c['predeclared_budgets'];inp=np.memmap(P/'actual_input_f32.bin',mode='r',dtype='<f4');out=np.memmap(P/'actual_output_f32.bin',mode='r',dtype='<f4');assert len(inp)==9*N+32 and len(out)==21*N+32
for a in (inp,out):assert (a[:16]==np.float32(91.25)).all() and (a[-16:]==np.float32(91.25)).all()
a=inp[16:-16].reshape(9,N);b=out[16:-16].reshape(21,N);maxima={k:0.0 for k in bounds if not k.startswith('actual_')};bad=0;worst={};minimum_input_norm=1e99;maximum_input_norm=0
for start in range(0,N,65536):
 stop=min(N,start+65536);A=a[:,start:stop].T.astype('f8').reshape(-1,3,3);O=b[:,start:stop].T.astype('f8');assert np.isfinite(A).all() and np.isfinite(O).all();U=O[:,:9].reshape(-1,3,3);S=O[:,9:12];V=O[:,12:].reshape(-1,3,3);norm=np.sqrt((A*A).sum((1,2)));denom=np.maximum(1,norm);sigma=np.linalg.svd(A,compute_uv=False);sigma_got=np.sort(np.abs(S),axis=1)[:,::-1]
 rec=(U*S[:,None,:])@np.swapaxes(V,1,2);ue=np.swapaxes(U,1,2)@U-np.eye(3);ve=np.swapaxes(V,1,2)@V-np.eye(3)
 metrics={'normalized_Frobenius_reconstruction_error':np.sqrt(((rec-A)**2).sum((1,2)))/denom,'normalized_sorted_absolute_singular_value_error':np.max(np.abs(sigma_got-sigma),axis=1)/denom,'U_orthogonality_Frobenius_error':np.sqrt((ue*ue).sum((1,2))),'V_orthogonality_Frobenius_error':np.sqrt((ve*ve).sum((1,2))),'det_U_or_V_distance_from_plus_one':np.maximum(abs(np.linalg.det(U)-1),abs(np.linalg.det(V)-1))}
 fail=np.zeros(stop-start,dtype=bool)
 for key,val in metrics.items():
  j=int(np.argmax(val));value=float(val[j]);fail|=val>bounds[key]
  if value>maxima[key]:maxima[key]=value;worst[key]={'matrix_index':start+j,'value':value,'bound':bounds[key]}
 bad+=int(fail.sum());minimum_input_norm=min(minimum_input_norm,float(norm.min()));maximum_input_norm=max(maximum_input_norm,float(norm.max()))
d={'all_1048576_complete_factorizations_checked':True,'all_22020096_raw_output_values_finite':True,'independent_LAPACK_singular_values':True,'source_svd_CPU_not_used':True,'budgets_declared_before_this_audit':True,'predeclared_contract_sha256':hashlib.sha256((P.parent/'predeclared_contract.json').read_bytes()).hexdigest(),'all_pass':bad==0,'failed_matrices':bad,'maxima':maxima,'worst':worst,'actual_input_norm_range':[minimum_input_norm,maximum_input_norm],'raw_output_identity_claim':False,'scope':'OriginalactualSoAdatasetnativeGPUfullfactorization diagnostic; nottargetverification, notoriginalweakCPUchecker, no universalcondition/dynamicrangeclaim.'};(P/'independent_full_dataset_audit.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
