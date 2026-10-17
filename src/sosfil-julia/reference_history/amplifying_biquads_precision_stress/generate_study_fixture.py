#!/usr/bin/env python3
"""Serial-section recurrence, independent of GPU time-skewed pipeline."""
from pathlib import Path
import math,struct,json,hashlib
P=Path(__file__).resolve().parent;prefix=[];gold=[];cases=[]
def cast(x,bits):return struct.unpack('<f',struct.pack('<f',x))[0] if bits==32 else float(x)
def cascade(x,sos,zi,signals,samples,advance=True):
 out=[]
 for signal in range(signals):
  v=x[signal*samples:(signal+1)*samples]
  for section in range(64):
   b0,b1,b2,a0,a1,a2=sos[6*section:6*section+6];z0,z1=zi[(signal*64+section)*2:(signal*64+section)*2+2]
   # Preserve every upstream startup transient: each stage after first receives
   # one extra leading zero, then the entire preceding stage's output history.
   if advance and section>0:v=[0.]+v
   w=[]
   for u in v:y=b0*u+z0;z0=b1*u-a1*y+z1;z1=b2*u-a2*y;w.append(y)
   v=w
  out.extend(v[63:] if advance else v)
 return out
for bits in [32,64]:
 for kind,signals,samples in [(0,2,65),(1,3,257)]:
  sos=([1.]*384 if kind==0 else [.875,.125,-.0625,1.,-.125,.0625]*64);zi=[1. if kind==0 else 0.]*(signals*128);x=[]
  for signal in range(signals):
   for j in range(samples):x.append(math.sin(2*3.14*(signal+1+j)) if kind==0 else (float(j==0) if signal==0 else .25 if signal==1 else ((j*17+3)%257-128)/256))
  sos=list(map(lambda v:cast(v,bits),sos));zi=list(map(lambda v:cast(v,bits),zi));x=list(map(lambda v:cast(v,bits),x));shared=64+len(zi)+384;prefix.extend([bits,signals,samples,64,2,shared]+sos+zi+x)
  v=x;phases=[]
  for step in range(1,6):
   v=cascade(v,sos,zi,signals,samples)
   if step in [1,2,5]:phases.extend(v)
  gold.extend(phases);standard=cascade(x,sos,zi,signals,samples,False);first=cascade(x,sos,zi,signals,samples);cases.append({'bits':bits,'signals':signals,'samples':samples,'sections':64,'shared_elements_allocated':shared,'shared_elements_used':576,'input_values':6+len(sos)+len(zi)+len(x),'output_values':3*len(x),'source_vs_unshifted_initial_state_max_error':max(abs(a-b) for a,b in zip(first,standard)),'effective_initial_state':'Each stage receives a leading zero plus the entire upstream startup history; discard63 pipeline latency outputs at final stage','kind':'source_allones_coefficients_zi1' if kind==0 else 'stable_biquads_zero_zi_impulse_constant_pattern'})
assert len(prefix)==4642 and len(gold)==5406
(P/'study_input_f64.bin').write_bytes(struct.pack('<4642d',*prefix));(P/'study_expected_f64.bin').write_bytes(struct.pack('<5406d',*gold));w={'benchmark':'sosfil','cases':cases,'prefix_count':len(prefix),'checked_output_values':len(gold),'artifact_count':len(prefix)+len(gold),'oracle':'Float64 serial-section recurrence with progressively prepended zero inputs including upstream transient history, final latency63 sliced; no GPU time-skewed indexing','timed_calls_per_case':3,'warmup_calls_before_roi':2,'captured_steps':[1,2,5],'source_default_scope':'Native30warmups/8signals/1000000samples unmeasured; kernel64sections unchanged, named2x65/3x257 and2warmups/3timed repeatedfilter applications','files':{}}
for f in ['study_input_f64.bin','study_expected_f64.bin','../sosfil-cuda/main.cu','../sosfil-cuda/reference.h']:
 b=(P/f).read_bytes();w['files'][f]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(P/'scientific_operator_witness.json').write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w,indent=2))
