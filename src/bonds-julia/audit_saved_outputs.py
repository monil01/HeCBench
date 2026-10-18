from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/home/imo/HeCBench_julia');B=R/'src/bonds-julia';S=R/'.porting-state/portable_julia';P=next(p for p in reversed(sorted((S/'runs').glob('*_bonds/result.json')))if json.loads(p.read_text()).get('matched_performance_valid'));d=json.loads(P.read_text());cs=json.loads((B/'cases.json').read_text())['cases'];fixture=np.memmap(B/'input_u32.bin',dtype='<u4',mode='r');oracle=np.memmap(B/'oracle_f64.bin',dtype='<f8',mode='r');GUARD=0x4a5a5a5a;FG=float(np.array([GUARD],dtype='<u4').view('<f4')[0]);POISON=-1234567.25;rows=[];perf=[]
for model,m in d['models'].items():
 for t in m['trials']:
  ar=t['output_artifact'];v=np.memmap(ar['saved_path'],dtype='<f8',mode='r');off=0;initial=[];details=[]
  for c in cs:
   fields=[];fi=c['io']
   for k,width in enumerate([16,16,4,4,1,9,1,1,1,1,1]):
    n=width*c['n'];a=v[off:off+n+32];off+=n+32;g=GUARD if k<7 else FG;assert np.all(a[:16]==g)and np.all(a[-16:]==g)
    if k<7:assert np.array_equal(a[16:-16],fixture[fi:fi+n]);fi+=n
    else:assert np.all(a[16:-16]==POISON)
    fields.append(a)
   initial.append(fields)
  assert off==55016533
  for c,pre in zip(cs,initial):
   n=c['n'];fields=[]
   for k,width in enumerate([16,16,4,4,1,9,1,1,1,1,1]):
    size=width*n;a=v[off:off+size+32];off+=size+32;g=GUARD if k<7 else FG;assert np.all(a[:16]==g)and np.all(a[-16:]==g)
    if k==0:assert np.array_equal(a[16:-16].reshape(n,16)[:,1:],pre[k][16:-16].reshape(n,16)[:,1:])
    elif k<7:assert np.array_equal(a,pre[k])
    fields.append(a[16:-16])
   forward=fields[0].reshape(n,16)[:,0];outputs=fields[7:]+[forward];wanted=oracle[c['oo']:c['oo']+5*n].reshape(5,n);errors=[]
   for col,(a,b)in enumerate(zip(outputs,wanted)):
    assert np.isfinite(a).all()and np.all(a!=POISON);e=np.abs(a-b);budget=.002+.0001*np.abs(b);assert np.all(e<=budget)
    if col<3:assert np.all(e<=.002)
    errors.append(dict(field=['dirty','accrued','clean','yield','curve_forward'][col],max_abs=float(e.max()),max_rel=float((e/np.maximum(1e-300,np.abs(b))).max()),max_budget_fraction=float((e/budget).max())))
   assert np.array_equal(outputs[3],forward);assert np.array_equal(outputs[0].astype('<f4')-outputs[1].astype('<f4'),outputs[2].astype('<f4'))
   # Use actual saved GPU bond-rate bits to construct the independent last-payment NPV.
   rawrate=pre[5][16:-16].reshape(n,9)[:,8].astype('<u4').view('<f4').astype(np.float64);npv=(100+50*rawrate)*np.exp(-np.log1p(forward/2)/180);res=float(np.max(np.abs(npv-wanted[0])));assert res<=.002
   details.append(dict(n=n,actual_input_metadata_readonly_bits_guards_and_initial_poison_exact=True,all_defined_outputs_written=True,actual_curve_forward_equals_actual_yield_exact=True,source_FP32_dirty_minus_accrued_equals_clean_exact=True,independent_F64_default_last_cashflow_NPV_max_abs_residual=res,errors=errors))
  assert off==len(v)
  with Path(ar['saved_path']).open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==ar['sha256']
  rows.append(dict(model=model,trial=t['trial'],sha256=ar['sha256'],cases=details))
  assert len(t['case_times_seconds'])==4 and abs(sum(t['case_times_seconds'])-t['time_seconds'])<1e-11
  for c,seconds in zip(cs,t['case_times_seconds']):perf.append(dict(model=model,trial=t['trial'],n=c['n'],time_seconds=seconds))
  del v
means=[]
for model in d['models']:
 for c in cs:
  samples=[r['time_seconds']for r in perf if r['model']==model and r['n']==c['n']and r['trial']in[2,3,4,5]];means.append(dict(model=model,n=c['n'],mean_seconds=sum(samples)/4,samples=samples))
 assert abs(sum(r['mean_seconds']for r in means if r['model']==model)-d['models'][model]['mean_seconds'])<1e-11
(B/'output_audit.json').write_text(json.dumps(dict(result=str(P.relative_to(R)),result_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),checked_artifacts=len(rows),artifact_words=110033066,artifact_bytes=880264528,input_prefix_words=55016533,raw_SHA_classes=len({r['sha256']for r in rows}),all_pass=True,scope='All4actualinputAoS/readonlymetadata/outputbuffers/guards, mutatedcurveforwardasactualFP32value plusall4sourcepricingoutputs; trueFloat64closedform source30/360 defaultone-final-paymentregime. No alteredsource/replacementroot/math, no generalnegativerate/longdated/financialconventioncertification. Hugeyields source1dayconditioning, maxabsfieldreportedwithunits notuniformaccuracy.',trials=rows),indent=2)+'\n');(B/'per_case_performance.json').write_text(json.dumps(dict(result=str(P.relative_to(R)),protocol='five_trials_last_four_v1',samples=perf,means=means),indent=2)+'\n');print('all20fulltyped/aos/NPV/price/forward PASS',len(rows),{m:x['mean_seconds']for m,x in d['models'].items()})
