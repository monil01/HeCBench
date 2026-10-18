"""Independent full fieldwise audit; auxiliary values are explicitly untimed."""
from pathlib import Path
import json,struct,hashlib,math
from generate_fixture import CASE_SPECS,GUARD,objective,budget
B=Path(__file__).resolve().parent;R=B.parents[1];p=R/'.porting-state/portable_julia/runs/20261018T051019463493Z_michalewicz/result.json';r=json.loads(p.read_text());raw=(B/'input_f32.bin').read_bytes();inputs=[z[0] for z in struct.iter_unpack('<f',raw)];N=sum(n*d+17 for n,d,_ in CASE_SPECS);COUNT=sum(n*(d+1)+34 for n,d,_ in CASE_SPECS);rows=[];issues=[];saved={};timings=[]
def bits(xs):return struct.pack('<'+'f'*len(xs),*xs)
for model,m in r['models'].items():
 saved[model]=[]
 for trial in m['trials']:
  a=trial['output_artifact'];data=Path(a['saved_path']).read_bytes();sha=hashlib.sha256(data).hexdigest();assert sha==a['sha256'];v=[z[0] for z in struct.iter_unpack('<f',data)];ok=len(v)==COUNT;po=0;oo=N;inp=0;points=0;maxerr=0.;caseoutputs=[];largest=0.
  for n,d,fixture in CASE_SPECS:
   host=inputs[inp:inp+n*d];expectedprefix=[GUARD]*4+host+[GUARD]*4+[GUARD]*4+[0.]+[GUARD]*4;ok &=bits(v[po:po+n*d+17])==bits(expectedprefix);out=v[oo:oo+n+8];mins=v[oo+n+8:oo+n+17];ok &=bits(out[:4]+out[-4:]+mins[:4]+mins[-4:])==bits([GUARD]*16);ys=out[4:-4];truth=[objective(host[i*d:(i+1)*d]) for i in range(n)];expectedmin=min(0.,min(truth));actualmin=0.
   for y,t in zip(ys,truth):
    e=abs(y-t);maxerr=max(maxerr,e);largest=max(largest,e/budget(t,d));ok &=math.isfinite(y) and e<=budget(t,d)
    if y<actualmin:actualmin=y
   ok &=bits([actualmin])==bits([mins[4]]) and math.isfinite(mins[4]) and abs(mins[4]-expectedmin)<=budget(expectedmin,d)
   if fixture==1:ok &=bits([mins[4]])==bytes(4) and all(y>0 for y in ys)
   caseoutputs.append((ys,mins[4],d));points+=n;inp+=n*d;po+=n*d+17;oo+=n+17
  row={'model':model,'trial':trial['trial'],'pass':bool(ok),'artifact_sha256':sha,'values':len(v),'pointwise_objectives':points,'max_abs_analytic_error':maxerr,'largest_pointwise_budget_fraction':largest,'timed_minimum_consistent_with_auxiliary_values':True if ok else None};rows.append(row);saved[model].append(caseoutputs)
  if not ok:issues.append(row)
  assert len(trial['path_times_seconds'])==12 and math.isclose(sum(trial['path_times_seconds']),trial['time_seconds'],rel_tol=1e-9)
 for i,(n,d,f) in enumerate(CASE_SPECS):
  ts=[t['path_times_seconds'][i] for t in m['trials']];timings.append({'model':model,'n':n,'dim':d,'fixture':f,'path':i,'trial_times_seconds':ts,'mean_last_four_seconds':sum(ts[1:])/4})
for model,trials in saved.items():
 for t,outputs in enumerate(trials):
  for ref in ['cuda','cuda_jl']:
   for path,((ys,minimum,d),(refys,refmin,_)) in enumerate(zip(outputs,saved[ref][t])):
    if not all(abs(a-b)<=6e-5*d+2e-5*max(abs(a),abs(b)) for a,b in zip(ys+[minimum],refys+[refmin])):issues.append({'model':model,'reference':ref,'trial':t+1,'path':path,'issue':'fieldwise dual budget'})
a={'result':str(p.relative_to(R)),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checked_artifacts':len(rows),'complete_values':COUNT,'exact_actual_input_prefix_values':N,'primary':'Actual timed seededGPUminimum, incumbent0 resetonce beforeR100, neverbetweenrepeats','auxiliary':'FullperpointactualGPUobjectivecaptures after timing, not originaltimedarrays','scope':'Finitecandidateincumbent min(0,objectives), not globaloptimizationcertificate; source[0,4] preserved','oracle':'IndependentFloat64 truepi/math.sin/fsum perpoint; predeclared dimdependent Float32errorbudget, notMPFRformalproof','checks':rows,'issues':issues,'mean_seconds':{m:v['mean_seconds'] for m,v in r['models'].items()}}
(B/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');(B/'path_timing_summary.json').write_text(json.dumps({'result':str(p.relative_to(R)),'configuration_id':r['configuration_id'],'rows':timings,'protocol':'Mean2-5,12casewindows reportedperR100; noresetbetweeninternalrepeats; no measuredatomicattemptcounts'},indent=2)+'\n');print(json.dumps({'artifacts':len(rows),'issues':issues,'max_analytic_error':max(z['max_abs_analytic_error'] for z in rows),'max_budget_fraction':max(z['largest_pointwise_budget_fraction'] for z in rows)}));assert len(rows)==20 and not issues
