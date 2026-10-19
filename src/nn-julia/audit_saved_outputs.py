"""Full saved-state geometry/permutation audit from actual readbacks, no oraclefile reuse."""
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent
F=Path(sys.argv[1])if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261019T162329555284Z_nn/result.json'
r=json.load(open(F));plan=json.load(open(P/'study_plan.json'));rows=[];bindings={}
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
assert r['matched_performance_valid']and r['reference_agreement']and all(r['variant_verified'].values())
for model,v in r['models'].items():
 assert v['provenance_valid']and len(v['trials'])==5
 for f,h in v['adapter_file_hashes'].items():assert sha(f)==h;bindings[f]=h
 for t in v['trials']:
  ar=t['output_artifact'];f=Path(ar['saved_path']);assert sha(f)==ar['sha256'];x=np.fromfile(f,dtype='<f8');assert len(x)==plan['artifact_count']and np.isfinite(x).all()
  ip=0;op=plan['input_prefix_count'];ma=mr=0.;case_rows=[]
  for c in plan['cases']:
   n,k=c['n'],c['k'];head=x[ip:ip+8];ip+=8;assert list(head[:3])==[n,k,c['repeats']];assert list(head[3:5])==list(head[5:7]);assert np.array([int(head[7])],dtype='<u4').view('<f4')[0]==n
   q=np.array(head[5:7],dtype='<u4').view('<f4').astype('f8');words=2*(n+16)
   before=x[ip:ip+words];after=x[ip+words:ip+2*words];ip+=2*words;assert np.array_equal(before,after)
   assert np.array_equal(before,before.astype('<u4').astype('f8'));points=before.astype('<u4').view('<f4').reshape(n+16,2)
   assert (points[:8]==np.float32(91.25)).all()and(points[n+8:]==np.float32(91.25)).all()
   delta=points[8:n+8].astype('f8')-q
   ref=np.sqrt(np.einsum('ij,ij->i',delta,delta))
   pre=x[op:op+n];post=x[op+n:op+2*n];ids=x[op+2*n:op+3*n];guards=x[op+3*n:op+3*n+32];op+=3*n+32
   assert np.array_equal(pre,pre.astype('f4').astype('f8'));err=np.abs(pre-ref);assert np.all(err<=1e-5+8*np.finfo('f4').eps*np.abs(ref));ma=max(ma,float(err.max()));mr=max(mr,float(np.max(err/np.maximum(np.abs(ref),1e-300))))
   assert np.array_equal(ids,np.floor(ids));idx=ids.astype('i8');assert np.array_equal(np.sort(idx),np.arange(n));assert np.array_equal(post,pre[idx]);expected=np.argsort(ref,kind='stable')[:k];assert np.array_equal(idx[:k],expected)
   # Independent analytic near-axis controls are exact and uniquely ranked.
   assert np.array_equal(post[:k],np.arange(k,dtype='f8')*.25)
   guard_bits=np.array([np.float32(91.25)],dtype='f4').view('u4')[0];assert(guards==guard_bits).all()
   case_rows.append({'n':n,'topk':k,'GPU_distances_checked':n,'CPU_postdistances_checked':n,'full_record_ID_permutation_checked':n,'topk_IDs':idx[:k].tolist(),'actual_query':q.tolist(),'max_abs_F64norm_error':float(err.max()),'squared_norm_method':'Float64dot(delta,delta),sqrt; fromactualinputreadback, nooraclefile'})
  assert ip==plan['input_prefix_count']and op==len(x)
  rows.append({'model':model,'trial':t['trial'],'artifact':{'path':str(f.relative_to(R)),'sha256':ar['sha256'],'bytes':f.stat().st_size},'cases':case_rows,'all_fields_checked':len(x),'max_abs_norm_error':ma,'max_rel_norm_error':mr,'correct':True})
out={'result':{'path':str(F.relative_to(R)),'sha256':sha(F)},'records':rows,'all20_pass':True,'actual_source_bindings':bindings,'unique_primary_artifact_SHA256':sorted({x['artifact']['sha256']for x in rows}),'verification_scope':'Everyactualdistance/sourceinput/guard andfullCPUrecordID permutation+distanceassociation, notoriginalweatherrecordtext ororiginalexternaldataset CLI. Nativehostswaprepaired; originalfullCLIwitness preserved separately.','independence':'RecomputedFloat64squarednorm viaNumPy fromactualGPUinputechoes, bijection/directpostdist=predist[ID], independentlysortedminimizers andexactanalyticalaxiscontrols. No sharedFloat64oraclefile read here; no claimindependenthumanreview.','environment':{'Python':sys.version,'NumPy':np.__version__},'raw_captures_not_to_commit':True}
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('NN_SAVED_FULL20_PASS',len(rows),len(bindings),out['unique_primary_artifact_SHA256'])
