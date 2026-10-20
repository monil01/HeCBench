#!/usr/bin/env python3
from pathlib import Path
import itertools,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;F=R/'.porting-state/portable_julia/runs/20261019T215956436446Z_kmc/result.json';d=json.load(open(F));plan=json.load(open(P/'study_plan.json'));assert d['matched_performance_valid']and d['reference_agreement']and all(d['variant_verified'].values());models=['cuda','cuda_jl','ka','jacc'];saved={};files=[]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
expected=np.memmap(P/'oracle_output_f64.bin',dtype='<f8',mode='r');budget=np.memmap(P/'oracle_budget_f64.bin',dtype='<f8',mode='r');pref=plan['prefix_count'];exact_ranges=[];off=pref
for c in plan['cases']:
 n=c['n'];exact_ranges.append((off,n,'same_source_normorder'));off+=n
 for row in range(n):exact_ranges.extend([(off+row*(n+32),16,'guard'),(off+row*(n+32)+16,n,'GEMVdots'),(off+row*(n+32)+16+n,16,'guard')])
 off+=n*(n+32)
 for row in range(n):exact_ranges.append((off+row*(n+1),1,'source_rowID'))
 off+=n*(n+1)
assert off==plan['primary_count']
for m in models:
 saved[m]=[]
 for t in d['models'][m]['trials']:
  a=t['output_artifact'];p=Path(a['saved_path']);assert p.stat().st_size==plan['primary_bytes'];assert sha(p)==a['sha256'];x=np.memmap(p,dtype='<f8',mode='r');assert np.all(np.isfinite(x));e=np.abs(x[pref:]-expected);assert np.all(e<=budget);saved[m].append(x);files.append({'model':m,'trial':t['trial'],'path':str(p),'sha256':a['sha256'],'bytes':p.stat().st_size})
comparisons=[]
for a,b in itertools.combinations(models,2):
 for t,(x,y)in enumerate(zip(saved[a],saved[b]),1):
  assert np.array_equal(x[:pref].view('<u8'),y[:pref].view('<u8'));maximum=0.
  for k in range(pref,len(x),1<<20):
   aa=x[k:k+(1<<20)];bb=y[k:k+(1<<20)];e=np.abs(aa-bb);assert np.all(e<=2e-5+2e-6*np.abs(bb));maximum=max(maximum,float(e.max(initial=0)))
  for start,count,kind in exact_ranges:assert np.array_equal(x[start:start+count].view('<u8'),y[start:start+count].view('<u8')),(a,b,t,kind,start)
  comparisons.append({'models':[a,b],'trial':t,'status':'pass','max_abs_error':maximum,'fullactualinputprefix_exact':True,'allGEMVdots_norms_rowIDs_guards_bitexact':True})
# Compare named final case with the actual original unchanged source cal_km execution.
base=pref+sum(c['output_count']for c in plan['cases'][:-1]);n=513;dp=saved['cuda'][0][base+n:base+n+n*(n+32)].reshape(n,n+32)[:,16:-16];k=saved['cuda'][0][base+n+n*(n+32):].reshape(n,n+1);D=R/'.porting-state/diagnostics/kmc/raw_capture';originaldp=np.fromfile(D/'actual_dots_f32.bin',dtype='<f4').reshape(n,n).astype(np.float64);originalk=np.fromfile(D/'actual_kernel_f64.bin',dtype='<f8').reshape(n,n+1);assert np.array_equal(dp,originaldp)and np.array_equal(k.view('<u8'),originalk.view('<u8'))
bindings=[]
for m,v in d['models'].items():
 for path,h in v['adapter_file_hashes'].items():
  f=Path(path);f=f if f.is_absolute()else Path(v['adapter_source_path'])/f;assert sha(f)==h;bindings.append({'model':m,'path':str(f),'sha256':h})
report={'benchmark':'kmc','status':'pass','result':str(F.relative_to(R)),'result_sha256':sha(F),'all6pairs_times5':comparisons,'independent_full_oracle_checked_all20':True,'original_unchanged_cal_km_same513case_dots_and_RBF_bitexact':True,'original_source_capture_timing_invalid':True,'count_each_trial':plan['primary_count'],'prefix_count':pref,'files':files,'source_binding_checks':len(bindings),'source_bindings':bindings,'numpy':np.__version__,'source_full6000_and_SVM_quality_not_certified':True}
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS all20independent/all30pairtrials; sourceoriginal513dots/RBFbitexact;',len(bindings),'bindings')
