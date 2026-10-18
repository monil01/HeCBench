"""CPU-only full artifact review with a separate exact-integer GEMM oracle."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=R/'.porting-state/portable_julia/runs/20261018T115029851782Z_wmma/result.json';r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert r['matched_performance_valid']and all(r['variant_verified'].values());input=np.fromfile(P/'input_u32.bin',dtype='<u4');oracle=np.fromfile(P/'oracle_f32.bin',dtype='<f4');prefix=[];integer_oracles=[]
for c in plan['cases']:
 m,n,k=c['m'],c['n'],c['k'];na=m*k;nb=n*k;nc=m*n;io=c['io'];chunk=input[io:io+na+nb+nc];prefix.extend(chunk);prefix.extend([c['impl'],m,n,k,100,32 if c['impl']==0 else 128,1 if c['impl']==0 else 4,16,16,16])
 A=chunk[:na].astype('<u2').view('<f2').astype('f8').reshape(m,k);B=chunk[na:na+nb].astype('<u2').view('<f2').astype('f8').reshape(n,k).T;C=chunk[na+nb:].view('<f4').astype('f8').reshape(m,n)
 Ai=np.rint(A*32).astype('i8');Bi=np.rint(B*16).astype('i8');Ci=np.rint(C*8).astype('i8');assert np.array_equal(A,Ai/32)and np.array_equal(B,Bi/16)and np.array_equal(C,Ci/8)
 numerator=Ai@Bi+256*Ci;expected=(numerator.astype('f8')/1024).astype('f4').ravel();assert np.array_equal(expected.view('u4'),oracle[c['oo']:c['oo']+nc].view('u4'));integer_oracles.append(expected)
prefix=np.array(prefix,dtype='<u4');wanted=np.concatenate(integer_oracles).view('<u4');guard=np.tile(np.r_[np.full(64,0x3555,dtype='u4'),np.full(64,0x4b123456,dtype='u4')],11);assert len(prefix)==2014062 and len(wanted)==673536
rows=[];bindings={};baseline=None
for model,s in r['models'].items():
 for p,h in s['adapter_file_hashes'].items():assert sha(p)==h;bindings[p]=h
 for t in s['trials']:
  p=Path(t['output_artifact']['saved_path']);a=np.fromfile(p,dtype='<u4');assert len(a)==2689006 and np.array_equal(a[:len(prefix)],prefix)and np.array_equal(a[len(prefix):len(prefix)+len(wanted)],wanted)and np.array_equal(a[-len(guard):],guard)
  if baseline is None:baseline=a
  assert np.array_equal(a,baseline)
  rows.append(dict(model=model,trial=t['trial'],path=str(p),sha256=sha(p),count=len(a),exact_prefix_count=len(prefix),full_output_count=len(wanted),guard_count=len(guard),bit_mismatches=0,int64_scaled_GEMM_oracle_matches=True,component_time_seconds=t['component_time_seconds']))
D=R/'.porting-state/portable_julia/diagnostics/wmma';raw=np.fromfile(D/'impl1_96x80_output_f32.bin',dtype='<f4').reshape(96,80);finite=np.isfinite(raw);mask=np.zeros_like(finite);mask[:64,:64]=True;assert np.array_equal(finite,mask)and np.isnan(raw[~mask]).all()
def source(n):
 a=np.arange(n,dtype='i8')%13;return np.where(a%3,-a,a)
A=source(96*16).reshape(96,16);B=source(80*16).reshape(80,16).T;C=source(96*80).reshape(96,80);exact=(.5*(A@B)+2*C).astype('f4');assert np.array_equal(raw[mask].view('u4'),exact[mask].view('u4'))
w=json.loads((D/'native_geometry.json').read_text());w.update(output_sha256=sha(D/'impl1_96x80_output_f32.bin'),output_bytes=(D/'impl1_96x80_output_f32.bin').stat().st_size,source_sha256=sha(R/'src/wmma-cuda/main.cu'),FP64_or_integer_GEMM_finite_subset_exact=True,defined_output_expected_count=7680,scientific_boundary='Valid64tileprimary shapes unaffected; originalhostvalidation is insufficient for all16multiples. No kernel/host correction made to native, no claim of originalfullCLI domain certification.',GPU_capture_kept_local_not_sourceGit=True)
(P/'native_geometry_witness.json').write_text(json.dumps(w,indent=2)+'\n')
out=dict(result=str(F.relative_to(R)),result_sha256=sha(F),CPU_only=True,no_GPU_rerun=True,NumPy=np.__version__,independent_oracle='ScaleA×32/B×16/C×8 to exactInt64; Dnumerator=Ai@Bi+256Ci, divide1024 thenF32. Separate from generatingFP64GEMM and originalCPUreference.',rows=rows,executed_source_bindings=bindings,source_geometry_diagnostic=w)
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('WMMA',len(rows),'captures',len(bindings),'bindings',len(wanted),'outputs, diagnosticNaNs',(~finite).sum())
