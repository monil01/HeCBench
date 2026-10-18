from pathlib import Path
import json,hashlib,re
import numpy as np
R=Path('/home/imo/HeCBench_julia');B=R/'src/tridiagonal-julia';S=R/'.porting-state/portable_julia';P=next(p for p in reversed(sorted((S/'runs').glob('*_tridiagonal/result.json')))if json.loads(p.read_text()).get('matched_performance_valid'));d=json.loads(P.read_text());cs=json.loads((B/'cases.json').read_text());f=np.fromfile(B/'input_f32.bin',dtype='<f4');oracle=np.fromfile(B/'oracle_f64.bin',dtype='<f8');rows=[];metrics={};stage_rows=[]
for model,mr in d['models'].items():
 for t in mr['trials']:
  ar=t['output_artifact'];v=np.memmap(ar['saved_path'],dtype='<f8',mode='r');off=0;casechecks=[];maxsol=0.;maxws=0.;maxratio=0.;maxback=0.
  for c in cs['states']:
   n=c['n'];m=c['m'];z=n*m
   for k in range(7):
    a=v[off:off+z+16];off+=z+16;assert np.all(a[:8]==1048575)and np.all(a[-8:]==1048575)
    want=f[c['io']+k*z:c['io']+(k+1)*z].astype(np.float64)if k<4 else np.full(z,-12345.5);assert np.array_equal(a[8:-8],want)
  assert off==2537136
  for c in cs['states']:
   n=c['n'];m=c['m'];z=n*m;mode=c['mode'];reordered=mode in[5,7,8];arr=[];localmaxsol=0.;localmaxws=0.;localratio=0.
   for k in range(7):
    a=v[off:off+z+16];off+=z+16;assert np.all(a[:8]==1048575)and np.all(a[-8:]==1048575);x=a[8:-8];want=oracle[c['oo']+k*z:c['oo']+(k+1)*z];diff=np.abs(x-want);numeric=(not reordered and k==4)or(reordered and k in[3,4])or(k==6 and mode in[4,5,8]);assert np.isfinite(x).all()
    if numeric:
     budget=2e-5+2e-4*np.abs(want);assert np.all(diff<=budget);localratio=max(localratio,float(np.max(diff/budget)))
     if k==6:localmaxws=max(localmaxws,float(diff.max()))
     else:localmaxsol=max(localmaxsol,float(diff.max()))
    else:assert np.array_equal(x,want)
    arr.append(x)
   x=arr[3 if reordered else 4].reshape(m,n);a,b,cval,rhs=[f[c['io']+k*z:c['io']+(k+1)*z].reshape(m,n).astype(np.float64)for k in range(4)];res=b*x-rhs;res[:,1:]+=a[:,1:]*x[:,:-1];res[:,:-1]+=cval[:,:-1]*x[:,1:];back=np.max(np.abs(res),axis=1)/(np.max(np.abs(a)+np.abs(b)+np.abs(cval),axis=1)*np.max(np.abs(x),axis=1)+np.max(np.abs(rhs),axis=1));assert np.all(back<=1e-5);maxback=max(maxback,float(back.max()));maxsol=max(maxsol,localmaxsol);maxws=max(maxws,localmaxws);maxratio=max(maxratio,localratio);casechecks.append(dict(case=c['case_id'],mode=mode,max_solution_abs_error=localmaxsol,max_elimination_workspace_abs_error=localmaxws,max_forward_budget_ratio=localratio,max_normwise_backward_error=float(back.max())))
  assert off==len(v);assert hashlib.sha256(Path(ar['saved_path']).read_bytes()).hexdigest()==ar['sha256'];rows.append(dict(model=model,trial=t['trial'],artifact_sha256=ar['sha256'],cases=casechecks));metrics.setdefault(model,[]).append(dict(trial=t['trial'],max_solution_abs_error=maxsol,max_elimination_workspace_abs_error=maxws,max_forward_budget_ratio=maxratio,max_normwise_backward_error=maxback));del v
 logs=[]
 for run in sorted((S/'runs').glob('*_tridiagonal')):
  log=run/(model+'_run1.log')
  if log.exists()and any(ar['output_artifact'].get('saved_path','').startswith(str(run))for ar in mr['trials']):logs.append(log)
 if len(logs)!=1:
  # Reused artifact paths may be source series; choose its explicit saved run.
  log=Path(mr['trials'][0]['output_artifact']['saved_path']).parent/(model+'_run1.log');assert log.exists();logs=[log]
 for line in logs[0].read_text().splitlines():
  match=re.match(r'STUDY_CASE trial=(\d+) case=(\d+) mode=(\d+) stage_seconds=(.*)',line)
  if match:
   tr,ci,mo=map(int,match.groups()[:3]);times=[float(x)for x in match[4].strip(',').split(',')];assert len(times)==(6 if mo in[5,7,8]else 1);stage_rows.append(dict(model=model,trial=tr,case=ci,mode=mo,stage_seconds=times,total_seconds=sum(times),stage_names=['transpose_a','transpose_b','transpose_c','transpose_d','solve','transpose_x']if len(times)==6 else['solve']))
assert len(stage_rows)==900
for model,mr in d['models'].items():
 for t in mr['trials']:
  total=sum(x['total_seconds']for x in stage_rows if x['model']==model and x['trial']==t['trial']);assert abs(total-t['time_seconds'])<=1e-11
out=dict(result=str(P.relative_to(R)),result_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),checked_saved_artifacts=len(rows),artifact_words=5074272,input_prefix_words=2537136,interpretation='ActualsavedGPUarrays: independentFractionThomas/fullphysicalsolution andeliminationworkspace +Float64normwisebackwarderrorforeachsystem/eachpath. Exactallinputecho/nonoutputphysicalstates/guards. BeforeeachstagewarmoutsideROI,3timedcalls+completion averaged; onlyfinalstageoutputs captured, not all internal repeatedwrites. Fivecheckedtrialsmean2-5, no three-trial measurement.',metrics=metrics,trials=rows);(B/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');(B/'stage_performance.json').write_text(json.dumps(dict(result=out['result'],protocol='five_trials_last_four_v1',stage_rows=stage_rows),indent=2)+'\n');print(out['result'],len(rows),len({x['artifact_sha256']for x in rows}),metrics)
