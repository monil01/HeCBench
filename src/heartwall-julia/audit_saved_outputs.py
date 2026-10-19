#!/usr/bin/env python3
"""Read-only complete saved primary and actual untimed replay comparison.
Bounded NumPy memory maps/chunks, all values, all 6 pairs; exact categorical/guards.
"""
from pathlib import Path
import json,hashlib,itertools,re,sys
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent
F=Path(sys.argv[1])if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261019T214127826428Z_heartwall/result.json'
d=json.load(open(F));assert d['matched_performance_valid']and d['reference_agreement']and all(d['variant_verified'].values())
B=json.load(open(P/'buffer_layout.json'))['buffers'];A=json.load(open(P/'artifact_layout.json'));base=102+A['final_state_count'];framevals=488096;step=1+framevals+A['final_state_count'];auxcount=base+11*step
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
models=['cuda','cuda_jl','ka','jacc'];primary={};aux={};files=[]
for m in models:
 v=d['models'][m];assert len(v['trials'])==5
 primary[m]=[]
 for t in v['trials']:
  a=t['output_artifact'];p=Path(a['saved_path']);assert p.stat().st_size==A['primary_bytes'];primary[m].append(np.memmap(p,dtype='<f4',mode='r'));files.append({'model':m,'kind':'primary','trial':t['trial'],'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 tracebase=R/f'.porting-state/porting_logs/heartwall-julia-shared/traces/{m}'
 paths=list(tracebase.glob('actual_all_frame_states_*.bin'));assert len(paths)==1
 p=paths[0];assert p.stat().st_size==auxcount*4;aux[m]=np.memmap(p,dtype='<f4',mode='r');files.append({'model':m,'kind':'actual_untimed_replay','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 # This is newly observed replay, not a timed intermediate capture.
 for k in range(11):assert aux[m][base+k*step]==k
report=[];CHUNK=1<<20
fieldmax={b['name']:0.0 for b in B}
def compare(x,y,start,n,exact=False):
 maximum=0.
 for k in range(0,n,CHUNK):
  aa=x[start+k:start+min(n,k+CHUNK)];bb=y[start+k:start+min(n,k+CHUNK)];assert np.all(np.isfinite(aa))and np.all(np.isfinite(bb))
  if exact:assert np.array_equal(aa.view('<u4'),bb.view('<u4'))
  else:
   a=aa.astype(np.float64);b=bb.astype(np.float64);e=np.abs(a-b);assert np.all(e<=.002+2e-6*np.abs(b));maximum=max(maximum,float(e.max(initial=0)))
 return maximum
def fields(x,y,offset):
 maximum=0.
 for b in B:
  start=offset+b['final_offset'];n=b['size']
  compare(x,y,start,16,True);compare(x,y,start+16+n,16,True)
  e=compare(x,y,start+16,n,b['type']=='int32');fieldmax[b['name']]=max(fieldmax[b['name']],e);maximum=max(maximum,e)
 return maximum
for a,b in itertools.combinations(models,2):
 maxima=[]
 for trial,(x,y)in enumerate(zip(primary[a],primary[b]),1):
  compare(x,y,0,A['prefix_count'],True);maximum=fields(x,y,A['prefix_count']);maxima.append(maximum)
 x,y=aux[a],aux[b];compare(x,y,0,base,True);am=[]
 for frame in range(11):
  off=base+frame*step;compare(x,y,off,1+framevals,True);am.append(fields(x,y,off+1+framevals))
 report.append({'models':[a,b],'primary_trials':5,'primary_max_abs_by_trial':maxima,'actual_untimed_frame_states':11,'auxiliary_max_abs_by_frame':am,'all_integer_fields_guards_input_prefixes_exact':True,'status':'pass'})
output={'benchmark':'heartwall','result':str(F),'result_sha256':sha(F),'status':'pass','primary_all6pairs_times5':report,'field_max_abs_error_all_saved_pairs':fieldmax,'primary_values_each':A['primary_count'],'auxiliary_values_each':auxcount,'actual_untimed_replay_not_timed_intermediates':True,'independent_CPU_checks':'Each actual process checked all11framecoordinates,templates,NCC/covariance/boxsum/guards; all30staticoperatororacles atfinal; fullotherintermediateglobalstates captured and crosscompared, notindependentlyclaimedall37intra-kerneloverwrittenstagevalues.','files':files,'numpy':np.__version__,'comparison_atol':.002,'comparison_rtol':2e-6,'exact_fields':'allactualinputs,allInt32bufferfields,allguards,frameIDs','clinical_accuracy_claim':False}
(P/'saved_output_audit.json').write_text(json.dumps(output,indent=2)+'\n');print('PASS all30primarypair/trials +all66actualreplaypair/frames; values/model',A['primary_count']*5+auxcount);print(fieldmax)
