"""CPU-only audit of immutable complete SpMM captures; no GPU work or retiming."""
from pathlib import Path
import json,sys,hashlib,re
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;F=Path(sys.argv[1]).resolve();r=json.loads(F.read_text());meta=json.loads((P/'fixture_provenance.json').read_text());input=np.fromfile(P/'input_u32.bin',dtype='<u4');stored=np.fromfile(P/'oracle_f64.bin',dtype='<f8');cases=meta['cases'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();prefix=[];expected=[];field_slices=[];start=0
for c in cases:
 m,k,n,z=c['m'],c['k'],c['n'],c['nnz'];o=c['input_offsets'];lens=[z,z,m+1,z,k*n];arr=[input[o[name]:o[name]+size]for name,size in zip(['rows','columns','row_offsets','values','B'],lens)];rows,cols,rp=arr[:3];val=arr[3].view('<f4');B=arr[4].view('<f4').reshape(k,n);assert rp[0]==0 and rp[-1]==z and np.all(np.diff(rp.astype('i8'))>=0);assert np.array_equal(np.repeat(np.arange(m),np.diff(rp.astype('i8'))),rows)and np.all(cols<k)
 A=np.zeros((m,k),dtype='f8');A[rows,cols]=val;oracle=A@B.astype('f8');alt=np.zeros_like(oracle)
 for row,col,v in zip(rows,cols,val):alt[row]+=float(v)*B[col].astype('f8')
 assert np.max(np.abs(alt-oracle))<1e-10;assert np.array_equal(oracle.ravel(),stored[c['oracle_offset']:c['oracle_offset']+m*n])
 for x in arr:prefix.append(x.astype('f8'))
 prefix.append(np.array([m,k,n,z,20,0,2,2,0,0,0,0,0x3f800000,0,2,0],dtype='f8'))
 for mode in ['COO','CSR']:expected.append(oracle.ravel());field_slices.append((c,mode,start,start+m*n));start+=m*n
prefix=np.concatenate(prefix);expected=np.concatenate(expected);assert len(prefix)==meta['prefix_count']and len(expected)==meta['output_count'];rows=[];bindings={};baseline=None
for model,v in r['models'].items():
 if v['status']=='missing':continue
 assert v['status']=='native_check_pass'and len(v['trials'])==5
 for name,h in v['adapter_file_hashes'].items():assert sha(name)==h;bindings[name]=h
 for t in v['trials']:
  a=t['output_artifact'];assert sha(a['saved_path'])==a['sha256'];raw=np.fromfile(a['saved_path'],dtype='<f8');assert len(raw)==meta['artifact_count']and raw[:len(prefix)].tobytes()==prefix.tobytes();out=raw[len(prefix):len(prefix)+len(expected)];assert np.isfinite(out).all()and np.array_equal(out,out.astype('f4').astype('f8'));assert np.all(raw[-meta['guard_count']:]==0x4e5a5a5a);diff=np.abs(out-expected);budget=1e-4+2e-5*np.abs(expected);assert np.all(diff<=budget)
  per=[]
  for c,mode,s,e in field_slices:
   block=out[s:e].reshape(c['m'],c['n']);assert np.all(block[c['zero_rows']]==0);per.append(dict(case=c['case'],format=mode,max_abs_error=float(np.max(diff[s:e])),max_normalized_budget_error=float(np.max(diff[s:e]/budget[s:e])),all_zero_rows_exact=True))
  dualmax=None
  if baseline is None:baseline=out.copy()
  else:dualmax=float(np.max(np.abs(out-baseline)));assert np.all(np.abs(out-baseline)<=1e-4+2e-5*np.abs(baseline))
  rows.append(dict(model=model,trial=t['trial'],artifact_sha256=a['sha256'],artifact_count=len(raw),exact_prefix_bytes=8*len(prefix),full_output_values=len(out),guard_words=meta['guard_count'],lossless_Float32_output_encoding=True,max_abs_error=float(diff.max()),max_normalized_budget_error=float(np.max(diff/budget)),max_abs_difference_from_first_native=dualmax,per_case_format=per))
lib=Path('/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140');lh=sha(lib);assert lh=='d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298';probes=[]
# Find immutable original execution logs even when all model rows were reused.
for model,v in r['models'].items():
 if v['status']=='missing':continue
 for run in sorted((R/'.porting-state/portable_julia/runs').glob('*_spgemm')):
  log=run/f'{model}_run1.log'
  if log.exists():
   text=log.read_text()
   if 'STUDY_TRIAL 'in text and 'STUDY_VENDOR function=cusparseSpMM'in text and str(lib)in text and 'version=12001'in text:
    probes.append(dict(model=model,log=str(log.relative_to(R)),log_sha256=sha(log),vendor_record=next(x for x in text.splitlines()if x.startswith('STUDY_VENDOR')),workspace_records=[x for x in text.splitlines()if x.startswith('STUDY_WORKSPACE')]));break
 assert any(x['model']==model for x in probes)
proof=dict(result_path=str(F.relative_to(R)),result_sha256=sha(F),NumPy=np.__version__,method='Rehash all actual captures/bindings; exact F64encoding of raw device UInt32inputbits/controlprefix; independently reconstructdenseAfromCOO+CSR equivalence and actualB; denseFP64GEMM and separate sparse-rowexpansion; fullFP32representableoutputs/correctzerorows/guardexact; all5dualcomparisons. NoGPUnewwork.',rows=rows,executed_source_bindings=bindings,actual_library={'path':str(lib),'sha256':lh,'version':12001,'actual_callsite_dladdr_and_version_logs':probes},limits=['Sameopaquevendorlibraryandalgorithm; zeroauthoredkernel/authorGPUJIT, frontendsorchestratelibrarycalls.','BaselinefirstsourceinitializerdrawcontrolledsameCOO/CSRinputs; originalseparateCSRdraws and4096scaleunmeasured.']);(P/'saved_output_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('AUDITED',len(rows),'fullcaptures',len(bindings),'bindings; maxCPUabs',max(x['max_abs_error']for x in rows),'actualDSOprobes',len(probes))
