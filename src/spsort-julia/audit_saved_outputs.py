"""CPU-only full immutable sparse-sort/permutation/gather audit."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;F=Path(sys.argv[1]).resolve();r=json.loads(F.read_text());meta=json.loads((P/'fixture_provenance.json').read_text());input=np.fromfile(P/'input_u32.bin',dtype='<u4');golden=np.fromfile(P/'oracle_u32.bin',dtype='<u4');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();prefix=[];expected=[];states=[]
for ci,c in enumerate(meta['cases']):
 m,n,z=c['m'],c['n'],c['nnz'];io=c['io'];rp=input[io:io+m+1];co=input[io+m+1:io+m+1+z];v=input[io+m+1+z:io+m+1+2*z];assert rp[0]==0 and rp[-1]==z and np.all(np.diff(rp.astype('i8'))>=0)and np.all(co<n)and np.isfinite(v.view('<f4')).all()
 # Independent Python row sorting (fixture uses NumPy argsort), with explicit key/payload association.
 p=[]
 for row in range(m):
  s,e=map(int,rp[row:row+2]);assert len(set(map(int,co[s:e])))==e-s;p.extend(sorted(range(s,e),key=lambda j:int(co[j])))
 perm=np.array(p,dtype='<u4');assert np.array_equal(np.sort(perm),np.arange(z,dtype='<u4'));record=np.r_[rp,co[perm],perm,v[perm]];assert np.array_equal(record,golden[c['oo']:c['oo']+len(record)]);expected.append(record);prefix.extend([rp,co,v,np.arange(z,dtype='<u4'),np.array([m,n,z,1000,32,0,0,0,0,1,5,16,ci,1,0,0],dtype='<u4')]);states.append((c,rp,co,v))
prefix=np.concatenate(prefix);expected=np.concatenate(expected);assert len(prefix)==meta['prefix_count']and len(expected)==meta['oracle_words'];rows=[];bindings={};baseline=None
for model,d in r['models'].items():
 if d['status']=='missing':continue
 assert d['status']=='native_check_pass'and len(d['trials'])==5
 for name,h in d['adapter_file_hashes'].items():assert sha(name)==h;bindings[name]=h
 for t in d['trials']:
  a=t['output_artifact'];assert sha(a['saved_path'])==a['sha256'];raw=np.fromfile(a['saved_path'],dtype='<u4');assert len(raw)==meta['artifact_count']and raw[:len(prefix)].tobytes()==prefix.tobytes();out=raw[len(prefix):len(prefix)+len(expected)];assert np.array_equal(out,expected)and np.all(raw[-meta['guard_count']:]==0x5a5a5a5a)
  if baseline is None:baseline=a['sha256']
  assert baseline==a['sha256'];pos=0;associations=0
  for c,rp,co,v in states:
   m,z=c['m'],c['nnz'];actualrp=out[pos:pos+m+1];pos+=m+1;sortedcol=out[pos:pos+z];pos+=z;perm=out[pos:pos+z];pos+=z;payload=out[pos:pos+z];pos+=z;assert np.array_equal(actualrp,rp)and np.array_equal(co[perm],sortedcol)and np.array_equal(v[perm],payload)and np.array_equal(np.sort(perm),np.arange(z,dtype='<u4'))
   for row in range(m):
    s,e=map(int,rp[row:row+2]);assert np.all((perm[s:e]>=s)&(perm[s:e]<e));assert np.all(sortedcol[s+1:e]>=sortedcol[s:e-1])if e>s else True
   associations+=z
  assert pos==len(out);rows.append(dict(model=model,trial=t['trial'],artifact_sha256=a['sha256'],count=len(raw),exact_prefix_bytes=4*len(prefix),full_output_words=len(out),guard_words=meta['guard_count'],all_rowptr_preserved=True,columns_sorted_each_row=True,permutation_complete_and_within_rows=True,value_bits_associations_checked=associations,internal_repeats=t['internal_repeats']))
lib=Path('/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140');lh=sha(lib);assert lh=='d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298';probes=[]
for model,d in r['models'].items():
 if d['status']=='missing':continue
 for run in sorted((R/'.porting-state/portable_julia/runs').glob('*_spsort')):
  log=run/f'{model}_run1.log'
  if log.exists():
   text=log.read_text();records=[x for x in text.splitlines()if x.startswith('STUDY_VENDOR')]
   if'STUDY_TRIAL 'in text and len(records)==2 and all(str(lib)in x and'version=12001'in x for x in records):
    probes.append(dict(model=model,log=str(log.relative_to(R)),log_sha256=sha(log),vendor_records=records,callsite_source_review='Nativefor-loopaddressesorderedXcsrsort/Gather; Juliaprintsexplicitsymbolnames; bothactualdladdrresolutionchecks/sourcehashbound.',workspace_records=[x for x in text.splitlines()if x.startswith('STUDY_WORKSPACE')]));break
 assert any(x['model']==model for x in probes)
proof=dict(result_path=str(F.relative_to(R)),result_sha256=sha(F),NumPy=np.__version__,method='Rehash every capture/binding; exact actualGPUinitialrowptr/unsortedcol/valuebits/identityP/controlprefix, fullCPUrowsortedrecord/240guards. IndependentPythonrowsortvsNumPyfixtureoracle; finalglobalPwithinrow/permutation/completeness andoriginalcolumn/valuebitassociation, allrows/entries notsamples. Source1000statefulsort/Gather calls retained; nodiagnosticGPUwork.',rows=rows,executed_source_bindings=bindings,identical_artifact_sha256=baseline,actual_library=dict(path=str(lib),sha256=lh,version=12001,callsite_probes=probes),limitations=['NativeR1000workloadlargelyalready-sortedafterfirstiteration; doesnotmeasurefreshunsorted-throughput.','DirectnamedCSRsyntheticinputnotoriginaldenseRNG; firstsourcegeometryretained, second41490caseunmeasured.','Zeroauthoreddevicekernels/vendorinteroponly.']);(P/'saved_output_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('AUDITED',len(rows),'captures',len(bindings),'bindings','fullassociationseach',sum(x['nnz']for x in meta['cases']))
