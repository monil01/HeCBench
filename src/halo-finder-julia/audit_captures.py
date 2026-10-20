"""CPU-only full capture/input-byte/physical-field audit; requires NumPy2.2.6.
No GPU launches, no modified tolerances, no claims about original N/S API fidelity.
"""
from pathlib import Path
import argparse,hashlib,json,math,struct
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture-root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--run-root',type=Path);a=p.parse_args();B=a.fixture_root;x=json.loads(a.result.read_text());assert x['matched_performance_valid']and all(x['variant_verified'].values())and x['reference_agreement']
def artifact_path(q):
 f=Path(q['saved_path']);return a.run_root/f.parent.name/f.name if a.run_root else f
cases=[];prefix=[]
for i,n,leaves in [(0,10001,1105),(2,31,4),(3,65,8)]:
 raw=(B/f'case{i}_input_Float32.bin').read_bytes();assert len(raw)==7*n*4;inp=np.frombuffer(raw,dtype='<f4').reshape(7,n);gold=np.fromfile(B/f'case{i}_oracle_Float64.bin',dtype='<f8').reshape(3,n);prefix.extend(np.frombuffer(raw,dtype='<u4').astype(np.float64));cases.append((i,n,leaves,inp,gold))
expected_prefix=np.asarray(prefix,dtype='<f8');assert len(prefix)==70679
reference_bytes=artifact_path(x['models']['cuda']['trials'][0]['output_artifact']).read_bytes();v=np.frombuffer(reference_bytes,dtype='<f8');assert len(v)==22436535 and np.array_equal(v[:70679],expected_prefix)
k=70679;input_spans=[(0,k)];force_blocks=[];final_blocks=[];case_frames=[]
def byte_values(q):
 assert np.isfinite(q).all()and ((q>=0)&(q<=255)&(q==np.rint(q))).all();return q.astype(np.uint32)
def u32(pos):
 q=byte_values(v[pos:pos+4]);assert len(q)==4;return int(q[0]|(q[1]<<8)|(q[2]<<16)|(q[3]<<24))
def ids(pos,count):
 q=byte_values(v[pos:pos+2*count]).reshape(count,2);return (q[:,0]|(q[:,1]<<8)).astype(np.int64)
for case_i,n,leaves,inp,gold in cases:
 seen=np.zeros(n,dtype=np.int64);before_case=k
 for leaf in range(leaves):
  start=k;header=[u32(k+4*i)for i in range(9)];k+=36;count,count1,gx,gy,bx,by,cut,soft,coeff=header
  assert 1<=count<=12 and count1==n and bx==32 and by==4 and gx==min((n+31)//32,32)and gy==min((count+3)//4,256)
  assert struct.unpack('<f',struct.pack('<I',cut))[0]==np.float32(np.float32(3.116326355)**2)
  assert struct.unpack('<f',struct.pack('<I',soft))[0]==np.float32(np.float32(.1)**2)and coeff==struct.unpack('<I',struct.pack('<f',1))[0]
  target=ids(k,count);k+=2*count;neighbor=ids(k,n);k+=2*n;assert ((target>=0)&(target<n)).all()and np.array_equal(np.sort(neighbor),np.arange(n));np.add.at(seen,target,1)
  assert (v[k:k+12*count]==0).all();k+=12*count;input_spans.append((start,k));force_blocks.append((k,count,case_i,target.copy()));k+=3*count
 assert (seen==1).all();assert u32(k)==n;input_spans.append((k,k+4));k+=4;order=[];indices=[]
 for j in range(n):
  id=int(ids(k,1)[0]);assert 0<=id<n;order.append(id);input_spans.append((k,k+2));k+=2;indices.extend(range(k,k+3));k+=3
 assert sorted(order)==list(range(n));final_blocks.append((case_i,n,np.asarray(order,dtype=np.int64),np.asarray(indices,dtype=np.int64).reshape(n,3)));case_frames.append({'fixture_index':case_i,'particles':n,'source_leaf_calls':leaves,'artifact_range':[before_case,k]})
assert k==len(v)
records=[];metrics=[];total_force_checks=0
for model,m in x['models'].items():
 assert len(m['trials'])==5 and m['complete']and m['timing_valid']and m['provenance_valid'];times=[q['time_seconds']for q in m['trials']];assert math.isclose(sum(times[1:])/4,m['mean_seconds'],rel_tol=1e-15)
 for t in m['trials']:
  q=t['output_artifact'];f=artifact_path(q);raw=f.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==q['sha256']and len(raw)==len(reference_bytes);w=np.frombuffer(raw,dtype='<f8');assert np.isfinite(w).all()
  # Strict byte equality for every input/metadata field, including the dynamic
  # source lists. No acceptance under a floating output tolerance for inputs.
  for start,stop in input_spans:assert raw[8*start:8*stop]==reference_bytes[8*start:8*stop],(model,t['trial'],start,stop)
  maximum_error=0;worst_budget_ratio=0;case_actual={}
  for pos,count,case_i,target in force_blocks:
   gold=next(c[4]for c in cases if c[0]==case_i);got=w[pos:pos+3*count].reshape(3,count);want=gold[:,target];err=np.abs(got-want);budget=.05+2e-4*np.abs(want);assert (err<=budget).all();maximum_error=max(maximum_error,float(err.max()));worst_budget_ratio=max(worst_budget_ratio,float((err/budget).max()));total_force_checks+=got.size
  for case_i,n,order,positions in final_blocks:
   inp=next(c[3]for c in cases if c[0]==case_i);gold=next(c[4]for c in cases if c[0]==case_i);actual=np.empty((3,n));actual[:,order]=w[positions].T;err=np.abs(actual-gold);assert (err<=.05+2e-4*np.abs(gold)).all();total_force_checks+=actual.size
   for pos,count,leaf_case,target in force_blocks:
    if leaf_case==case_i:assert np.array_equal(w[pos:pos+3*count].reshape(3,count),actual[:,target])
   net=actual.sum(axis=1);scale=np.abs(actual).sum();torque=np.cross(inp[:3].astype(np.float64).T,actual.T).sum(axis=0);torque_scale=(np.abs(inp[:3].astype(np.float64)).sum(axis=0)*np.abs(actual).sum(axis=0)).sum();assert (np.abs(net)<=1e-5*scale).all()and(np.abs(torque)<=1e-5*torque_scale).all()
   metrics.append({'model':model,'trial':t['trial'],'fixture_index':case_i,'particles':n,'full_field_max_abs_error':float(err.max()),'relative_L2_error':float(np.linalg.norm(actual-gold)/np.linalg.norm(gold)),'max_error_over_declared_combined_budget':float((err/(.05+2e-4*np.abs(gold))).max()),'net_force_over_sumabs':(net/scale).tolist(),'net_torque_over_sumabs_rF':(torque/torque_scale).tolist(),'raw_final_and_leaf_outputs_equal':True})
  assert t['correct']and t['checked_values']==30291 and t['scientific_GPU_dispatches']==1117 and t['case_leaves']==[1105,4,8]
  records.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':sha,'bytes':len(raw),'full_input_metadata_spans_compared_exactly':len(input_spans),'all_actual_input_encodings_strict_byte_pass':True,'full_leaf_and_final_fields_independently_checked':60582,'max_abs_error':maximum_error,'maximum_error_over_declared_combined_budget':worst_budget_ratio})
out={'all_pass':True,'result_path':str(a.result),'result_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'artifacts':20,'values_each':len(v),'actual_global_input_prefix_values':70679,'dynamic_source_input_encoding':'ExactUInt16IDs(two0..255bytes),mappedtofullcapturedFloat32position/massbits afteractualGPUecho; everylistcompleteallN andtargetsexactlyonce.','all_dynamic_input_and_metadata_bytes_equal_across_all20':True,'full_artifact_hash_checks':records,'independent_force_checks_total':total_force_checks,'physical_accuracy_per_case':metrics,'case_layout':case_frames,'guard_evidence':'SourceboundC exactmappedhostguardchecks before/aftereveryGPUlaunchandonfree. Guardbytes are not redundantly serialized; this audit doesnotclaim rawguardbytecapture.','source_known_API_defect_retained':True,'no_original_Newton_or_SR_fidelity_claim':True,'audit_engine':{'Python':'3.12','NumPy':np.__version__,'scope':'CPU-onlyindependentaudit; primaryverifieruseditsunchangedPythonstructengine, OS/sharedpackages notmodified.'}}
tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(a.output);print(json.dumps({'all_pass':True,'artifacts':20,'full_force_checks':total_force_checks,'relative_L2_range':[min(m['relative_L2_error']for m in metrics),max(m['relative_L2_error']for m in metrics)]}))
