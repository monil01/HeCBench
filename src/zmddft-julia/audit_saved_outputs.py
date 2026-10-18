"""CPU-only complete saved-evidence scan, independent Fourier controls, no GPU work."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=R/'.porting-state/portable_julia/runs/20261018T121317524539Z_zmddft/result.json';r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());N=33554432;PREFIX=N+520;STEP=524288
assert r['matched_performance_valid']and all(r['variant_verified'].values())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for a in iter(lambda:f.read(4*1024*1024),b''):h.update(a)
 return h.hexdigest()
input=np.memmap(P/'input_complex_f64.bin',dtype='<f8',mode='r');tw=np.fromfile(P/'twiddles_f64.bin',dtype='<f8');oracles=[np.memmap(P/f'oracle_stage{i}_f64.bin',dtype='<f8',mode='r')for i in range(3)]
expectedparams=np.array([256,256,256,100,64,16384,3,512],dtype='<f8');bindings={}
for s in r['models'].values():
 for p,h in s['adapter_file_hashes'].items():
  if p not in bindings:assert sha(p)==h;bindings[p]=h
rows=[];controls={}
def fourier_controls(path):
 y=np.memmap(path,dtype='<c16',mode='r',offset=(PREFIX+2*N)*8,shape=(256,256,256));x=input.view('c16').reshape(256,256,256)
 output_energy=0.;input_energy=0.
 flat=y.ravel();xin=x.ravel()
 for i in range(0,len(flat),STEP):
  a=flat[i:i+STEP];b=xin[i:i+STEP];output_energy+=float(np.sum(a.real*a.real+a.imag*a.imag));input_energy+=float(np.sum(b.real*b.real+b.imag*b.imag))
 parseval_rel=abs(output_energy-256**3*input_energy)/(256**3*input_energy)
 dc=complex(y[0,0,0]);expectedDC=complex(np.sum(x));dcerror=abs(dc-expectedDC)
 inv=np.fft.ifftn(y);roundtrip=float(np.max(np.abs(inv-x)));del inv
 roots=np.exp(-2j*np.pi*np.arange(256)/256);freqs=[]
 for a,b,c in [(0,0,0),(1,2,3),(17,29,41),(128,0,127)]:
  pa=roots[(a*np.arange(256))%256];pb=roots[(b*np.arange(256))%256];pc=roots[(c*np.arange(256))%256]
  direct=complex(np.dot(pa,np.einsum('ab,b->a',np.einsum('abc,c->ab',x,pc),pb)));actual=complex(y[a,b,c]);er=np.array([abs(actual.real-direct.real),abs(actual.imag-direct.imag)]);budget=1e-9+1e-12*np.abs([direct.real,direct.imag]);passed=bool(np.all(er<=budget));freqs.append(dict(frequency=[a,b,c],actual=[actual.real,actual.imag],direct_separable_DFT=[direct.real,direct.imag],max_component_abs_error=float(max(er)),max_component_normalized_error=float(max(er/budget)),passed=passed))
 assert parseval_rel<1e-12 and dcerror<=1e-9+1e-12*abs(expectedDC)and roundtrip<1e-12 and all(a['passed']for a in freqs)
 return dict(Parseval_relative_error=parseval_rel,DC_actual=[dc.real,dc.imag],DC_expected=[expectedDC.real,expectedDC.imag],DC_abs_error=dcerror,inverseFFT_max_complex_abs_error=roundtrip,independent_direct_DFT_coefficients=freqs,scope='CPU-only controls on fullfinalfield; inverseFFT/Parseval/DCplus directcomplexphasorcontractions (not sourcebutterflytranscription).')
for model,s in r['models'].items():
 for t in s['trials']:
  path=Path(t['output_artifact']['saved_path']);h=sha(path);a=np.memmap(path,dtype='<f8',mode='r');assert len(a)==134218312
  for i in range(0,N,STEP):assert np.array_equal(a[i:i+STEP].view('u8'),input[i:i+STEP].view('u8'))
  assert np.array_equal(a[N:N+512].view('u8'),tw.view('u8'))and np.array_equal(a[N+512:PREFIX],expectedparams)
  stage_errors=[]
  for stage,o in enumerate(oracles):
   ma=mb=0.;v=a[PREFIX+stage*N:PREFIX+(stage+1)*N]
   for i in range(0,N,STEP):
    actual=v[i:i+STEP];expected=o[i:i+STEP];diff=np.abs(actual-expected);budget=1e-9+1e-12*np.abs(expected);assert np.isfinite(actual).all()and np.all(diff<=budget);ma=max(ma,float(np.max(diff)));mb=max(mb,float(np.max(diff/budget)))
   stage_errors.append(dict(stage=stage,max_abs_error=ma,max_normalized_budget_error=mb,all_values_checked=N))
  assert np.all(a[-64:]==12345.25)
  if h not in controls:controls[h]=fourier_controls(path)
  rows.append(dict(model=model,trial=t['trial'],path=str(path),sha256=h,bytes=path.stat().st_size,count=len(a),exact_input_prefix_bytes=PREFIX*8,full_stage_values=3*N,guard_values=64,stage_errors=stage_errors,fourier_controls_key=h,source_provenance_valid=s['provenance_valid']))
D=R/'.porting-state/portable_julia/diagnostics/zmddft';impulses=[]
for model in ['cuda','cuda_jl','ka','jacc']:
 d=json.loads((D/f'{model}_impulse.json').read_text());path=Path(d['stage_artifact']);a=np.memmap(path,dtype='<f8',mode='r');assert len(a)==3*N;errs=[]
 for stage in range(3):
  o=np.memmap(P/f'impulse_stage{stage}_f64.bin',dtype='<f8',mode='r');v=a[stage*N:(stage+1)*N];ma=0.
  for i in range(0,N,STEP):diff=np.abs(v[i:i+STEP]-o[i:i+STEP]);assert np.isfinite(v[i:i+STEP]).all()and np.all(diff<=1e-9+1e-12*np.abs(o[i:i+STEP]));ma=max(ma,float(np.max(diff)))
  errs.append(dict(stage=stage,max_abs_error=ma,all_values_checked=N))
 d.update(artifact_sha256=sha(path),artifact_bytes=path.stat().st_size,independent_saved_stage_review=errs);impulses.append(d)
out=dict(result=str(F.relative_to(R)),result_sha256=sha(F),CPU_only=True,no_GPU_reruns=True,NumPy=np.__version__,chunk_values=STEP,rows=rows,unique_full_artifact_Fourier_controls=controls,analytic_impulse_controls=impulses,executed_source_bindings=bindings,scope='Every20primaryartifactall3full256³complexfields/inputprefix/guards independentlyscanned; every4untimedimpulseall3stagecapturevalueschecked. FourierscientificcontrolsreusedonlyafterindependentfullfileSHAprovesbyteidentity, notGPUtrialsreused.')
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('ZMDDFT',len(rows),'primarycaptures',len(impulses),'fullimpulsecaptures',len(bindings),'bindings',len(controls),'uniqueFouriercontrolfields')
