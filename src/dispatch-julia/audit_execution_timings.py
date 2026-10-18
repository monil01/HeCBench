"""Read-only timing/source audit for an actually empty kernel; no numeric oracle."""
from pathlib import Path
import hashlib,json,math,statistics,sys
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
rp=Path(sys.argv[1]) if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261018T053228734981Z_dispatch/result.json'
r=json.loads(rp.read_text());assert r['numerical_agreement_kind']=='no_numerical_outputs_execution_only' and r['matched_performance_valid'];records=[];sources=[];samples=0
for m,v in r['models'].items():
 assert len(v['trials'])==5 and v['provenance_valid'] and v['timing_valid']
 for t in v['trials']:
  assert t['correct'] and t['source_empty_kernel'] and t['execution_checked'] and t['checked_values']==0 and not t.get('output_artifact') and not t.get('numeric_outputs') and t['max_abs_error'] is None and t['max_rel_error'] is None
  assert t['completed_source_loop_launch_calls']==1102200
  modes={}
  for raw_name,out_name,batch in [('enqueue','enqueue',1),('single_dispatch','single_dispatch',1),('batch1000','batch1000_per_launch',1000)]:
   x=t['internal_timing_samples_seconds'][raw_name];assert len(x)==1100 and all(math.isfinite(v) and v>0 for v in x);samples+=len(x);retained=x[100:];mean=statistics.mean(retained)/batch;dev=statistics.pstdev(retained)/batch
   assert math.isclose(mean,t['component_time_seconds'][out_name],rel_tol=1e-9,abs_tol=1e-15)
   assert math.isclose(dev,t['component_stddev_seconds'][out_name],rel_tol=1e-9,abs_tol=1e-15)
   if batch==1000:assert math.isclose(mean,t['time_seconds'],rel_tol=1e-9,abs_tol=1e-15)
   modes[raw_name]=dict(raw_samples=len(x),source_warmup_excluded=100,retained_samples=1000,batch=batch,mean_seconds_per_launch=mean,stddev_seconds_per_launch=dev)
  records.append(dict(model=m,trial=t['trial'],numerical_agreement_claim=False,checked_numeric_values=0,execution_checked=True,source_loop_host_launch_calls=1102200,modes=modes))
 for f,h in v['adapter_file_hashes'].items():
  f=Path(f);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;sources.append(dict(model=m,path=str(f),sha256=h))
for role in ['ka','jacc']:
 assert all(x['status']=='execution_only_pass' and x['numerical_agreement_claim'] is False for ref in r['comparisons'][role].values() for x in ref['trials'])
result=dict(result=str(rp.relative_to(R)),validation_kind='no_numerical_outputs_execution_only',numerical_agreement_claim=False,numerical_accuracy='not_applicable_source_empty',numeric_artifacts_expected=0,records=records,raw_timing_samples_audited=samples,measured_source_bindings=sources,rounding_tolerance='Recomputemean/stddevfrom12significantdigitstoredrawtimings:rel1e-9abs1e-15statisticalroundingonly, notnumericalkerneloutputtolerance',host_launch_calls_all5_per_model=5511000,limits=['Sourcecompleted-loophostcalls, notindependentprofilercounterofactualGPUblocks.','No FLOPs/bytes/rooflinecoordinates orscientificaccuracyinferredfromemptykernel.'])
(P/'execution_timing_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(dict(execution_records=len(records),raw_timing_samples=samples,source_bindings=len(sources)))
