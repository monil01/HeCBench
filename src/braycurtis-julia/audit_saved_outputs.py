"""Vectorized independent mathematics from full actual device-input readbacks, not oracle assets."""
from pathlib import Path
import json,hashlib,sys,platform
import numpy as np
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=Path(sys.argv[1])if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261019T170239306493Z_braycurtis/result.json';r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());cache={};bindings={};records=[]
for model,m in r['models'].items():
 assert m['status']=='native_check_pass'and m['provenance_valid']and len(m['trials'])==5
 for name,sha in m['adapter_file_hashes'].items():
  p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;bindings[str(p.resolve().relative_to(R))]=sha
 for t in m['trials']:
  a=t['output_artifact'];p=Path(a['saved_path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==a['sha256'];assert len(raw)==plan['artifact_count']*8;x=np.frombuffer(raw,dtype='<f8');assert np.isfinite(x).all();prefix_sha=hashlib.sha256(raw[:plan['input_prefix_count']*8]).hexdigest();reused=prefix_sha in cache
  if not reused:
   offset=0;expected=[]
   for c in plan['cases']:
    dtype,b,mn,n,k=c['dtype'],c['b'],c['m'],c['n'],c['k'];h=x[offset:offset+11];offset+=11;assert np.array_equal(h,[dtype,b,mn,n,k,10,b,mn,n,k,10]);arrays=[]
    for count in [c['x1_count'],c['x2_count'],c['pairs']]*2:
     z=x[offset:offset+count+16];offset+=count+16;assert np.all(z[:8]==91.25)and np.all(z[-8:]==91.25);arrays.append(z[8:-8])
    for before,after in zip(arrays[:3],arrays[3:]):assert before.tobytes()==after.tobytes()
    A=arrays[0].reshape(b,mn,k);B=arrays[1].reshape(b,n,k);go=arrays[2].reshape(b,mn,n);delta=A[:,:,None,:]-B[:,None,:,:];plus=A[:,:,None,:]+B[:,None,:,:];assert np.all(delta!=0)and np.all(plus!=0);D=np.sum(np.abs(delta),axis=3);S=np.sum(np.abs(plus),axis=3);assert np.all(S>0);distance=D/S
    if c['domain']=='strictly_positive_noncusp':assert np.all((distance>=0)&(distance<=1))
    # Vectorized quotient derivative, algebraically independent fromsourceleft-associated perthreadloops.
    S4=S[:,:,:,None];D4=D[:,:,:,None];weights=go[:,:,:,None];g1=np.sum(weights*(np.sign(delta)*S4-np.sign(plus)*D4)/(S4*S4),axis=2);g2=np.sum(weights*(-np.sign(delta)*S4-np.sign(plus)*D4)/(S4*S4),axis=1)
    expected.append((c,[distance.ravel(),D.ravel(),S.ravel(),g1.ravel(),g2.ravel()]))
   assert offset==plan['input_prefix_count'];cache[prefix_sha]=expected
  offset=plan['input_prefix_count'];checks=[]
  for c,fields in cache[prefix_sha]:
   metrics=[]
   for name,expected in zip(['distance','absdiff_components','absplus_components','gradient_x1','gradient_x2'],fields):
    count=len(expected);z=x[offset:offset+count+16];offset+=count+16;assert np.all(z[:8]==91.25)and np.all(z[-8:]==91.25);actual=z[8:-8];at,rt=(1e-5,1e-4)if c['dtype']==32 else(1e-12,1e-10);error=np.abs(actual-expected);budget=at+rt*np.abs(expected);assert np.all(error<=budget),(model,c['id'],name,float(np.max(error/budget)));metrics.append(dict(field=name,full_values_checked=count,max_abs_error=float(np.max(error)),max_rel_error=float(np.max(error/np.maximum(np.abs(expected),1e-300))),max_fraction_of_declared_budget=float(np.max(error/budget)),atol=at,rtol=rt))
   checks.append(dict(case=c['id'],dtype=c['dtype'],metrics=metrics))
  assert offset==len(x);records.append(dict(model=model,trial=t['trial'],artifact=dict(path=str(p.relative_to(R)),sha256=a['sha256'],bytes=len(raw)),actual_prefix_sha256=prefix_sha,full_artifact_rehashed=True,mathematical_oracle_reused_after_exact_prefix_rehash=reused,complete_values_checked=len(x),cases=checks))
assert len(records)==20
(P/'saved_output_audit.json').write_text(json.dumps(dict(result=dict(path=str(F.relative_to(R)),sha256=hashlib.sha256(F.read_bytes()).hexdigest()),all20_pass=True,records=records,actual_source_bindings=bindings,unique_artifact_sha256=sorted({z['artifact']['sha256']for z in records}),independence='VectorizedNumPyFloat64 norm/signedquotient-gradient mathematics fromactualdeviceinputprefix; nooracleassetreads; exactbeforeafter/guards/domainchecked. Onlymathematicalcalculationcachedafterfullprefixrehasheachcapture.',environment=dict(python=platform.python_version(),numpy=np.__version__),raw_captures_not_to_commit=True),indent=2)+'\n');print('BRAY_FULL20_VECTOR_MATH_PASS',len(records),len(bindings),len(cache))
