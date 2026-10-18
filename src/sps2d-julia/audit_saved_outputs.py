"""Independent CSR structural/coordinate reconstruction of all20actualfullcaptures."""
from pathlib import Path
import json,sys,hashlib,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];F=Path(sys.argv[1]).resolve();r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());SENT=0x4e5a5a5a
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];bindings={};canonical=None
for model in ['cuda','cuda_jl','ka','jacc']:
 m=r['models'][model];assert len(m['trials'])==5 and m['status']=='native_check_pass'
 for p,h in m['adapter_file_hashes'].items():assert sha(p)==h;bindings[p]=h
 log=F.parent/(model+'_run1.log')
 # Reused models' actual logs are in their original execution bundle, never pretend new GPU executions.
 if not log.exists():
  for d in sorted((R/'.porting-state/portable_julia/runs').glob('*_sps2d')):
   p=d/(model+'_run1.log')
   if p.exists() and any(t['output_artifact']['saved_path'].startswith(str(d))for t in m['trials']):log=p;break
 text=log.read_text();assert '/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140' in text and 'version=12001' in text and 'index64base0 rowmajor alg0' in text
 for t in m['trials']:
  file=Path(t['output_artifact']['saved_path']);raw=np.fromfile(file,dtype='<u4');assert len(raw)==plan['artifact_words'];pos=0;cs=[]
  for c in plan['cases']:
   M,N,nnz=c['m'],c['n'],c['nnz'];rp=raw[pos:pos+2*(M+1)].view('<i8');pos+=2*(M+1);col=raw[pos:pos+2*nnz].view('<i8');pos+=2*nnz;bits=raw[pos:pos+nnz];val=bits.view('<f4');pos+=nnz;assert np.array_equal(raw[pos:pos+12],[M,N,nnz,1000,3,3,0,0,2,0,0,0]);pos+=12
   assert rp[0]==0 and rp[-1]==nnz and np.all(np.diff(rp)>=0) and np.all((col>=0)&(col<N)) and np.isfinite(val).all() and np.all((val>=1)&(val<2))
   dense=np.zeros(M*N,dtype='<u4')
   for i in range(M):
    a,b=map(int,rp[i:i+2]);assert np.all(np.diff(col[a:b])>0);dense[i*N+col[a:b]]=bits[a:b]
   cs.append((c,dense,rp,col,val))
  assert pos==plan['input_prefix_count'];invariants=[]
  for c,want,rp,col,val in cs:
   M,N,nnz=c['m'],c['n'],c['nnz'];actual=raw[pos:pos+M*N];pos+=M*N;assert np.array_equal(actual,want);v=actual.view('<f4').reshape(M,N);assert np.isfinite(v).all() and np.count_nonzero(v)==nnz
   # Exact dyadic sum/Frobenius controls independent of vectorized coordinate scatter.
   dense_sum=float(v.sum(dtype='f8'));sparse_sum=float(val.sum(dtype='f8'));dense_norm=float(np.square(v.astype('f8')).sum());sparse_norm=float(np.square(val.astype('f8')).sum());assert dense_sum==sparse_sum and dense_norm==sparse_norm
   invariants.append(dict(case=c['id'],m=M,n=N,nnz=nnz,empty_rows=int(np.count_nonzero(np.diff(rp)==0)),full_dense_values=M*N,sum_exact=dense_sum,frobenius_squared_exact=dense_norm,coordinates_sorted_unique=True))
  assert np.all(raw[pos:]==SENT) and len(raw)-pos==384;digest=sha(file)
  if canonical is None:canonical=digest
  assert digest==canonical,'all4fullbitcaptures must be identical'
  rows.append(dict(model=model,trial=t['trial'],path=str(file.relative_to(R)),sha256=digest,full_values_checked=len(raw),exact_input_prefix_words=plan['input_prefix_count'],full_Float32_output_values=plan['oracle_words'],guards_exact_words=384,CSR_independent_invariants=invariants,zero_authored_device_kernels=True,vendor_provenance_log=str(log.relative_to(R)),component_seconds=t['component_time_seconds']))
assert len(rows)==20 and r['matched_performance_valid']and r['reference_agreement']and all(r['variant_verified'].values())
(P/'saved_output_audit.json').write_text(json.dumps(dict(correct=True,rows=rows,executed_source_bindings=bindings,actual_vendor_library=dict(path='/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140',sha256=sha(Path('/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140')),version=12001),all20full_artifacts_bitidentical=True,source_device_kernel_definitions=0,vendor_internal_kernel_count_and_compiler_passes=None,scope='libraryconversionoperatorfidelity/numericalexactness, notcustomframeworkGPUkernelperformance'),indent=2)+'\n');print('SPS2D_SAVED_AUDIT',len(rows),len(bindings),canonical,flush=True)
