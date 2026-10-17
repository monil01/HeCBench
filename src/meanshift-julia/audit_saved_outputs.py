#!/usr/bin/env python3
"""Independent full saved-position/error audit; no GPU execution or timing."""
from pathlib import Path
import array,hashlib,json,math,struct,sys
P=Path(__file__).resolve().parent;R=P.parent.parent;expected=array.array('d');expected.frombytes((P/'study_expected_f64.bin').read_bytes());host=struct.unpack('<30144f',(P/'study_input_f32.bin').read_bytes());prefix=struct.pack('<30144d',*host);out={'metric_definition_erratum':{'raw_custom_Julia_max_rel_error':'max_absolute_error / atol, a scaled absolute residual, not mathematical relative error. Preserve raw fields/source; use corrected metrics below.','native_field':'maximum per-element error/max(abs(expected),atol,1e-300).','corrected_metric':'maximum abs(got-expected)/max(abs(expected),0.01); CPU Double recurrence outputs only, prefix excluded.','acceptance_unchanged':'CPU/source/dual acceptance uses absolute0.01 and exact prefix; no pass or timing changes.'},'configurations':{}}
for arg in sys.argv[1:]:
 run=Path(arg);run=run if run.is_absolute()else R/run;r=json.loads(run.read_text());assert r['matched_performance_valid'];records={};reference=None
 for model,x in r['models'].items():
  rows=[]
  for q in x['trials']:
   f=Path(q['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;got=array.array('d');got.frombytes(raw);got=got[30144:];assert len(got)==30000 and all(map(math.isfinite,got));err=max(abs(a-b)for a,b in zip(got,expected));rel=max(abs(a-b)/max(abs(b),.01)for a,b in zip(got,expected));assert err<=.01
   clustermax=max(sum((got[3*i+j]-100*(i%3))**2 for j in range(3))for i in range(10000));assert clustermax<=10
   if reference is None:reference=raw
   assert raw==reference
   rows.append({'trial':q['trial'],'true_CPU_max_abs_error':err,'true_CPU_max_relative_error_with_atol_floor':rel,'original_max_squared_cluster_center_distance':clustermax,'all4all5_artifact_bytes_identical':True,'input_prefix_exact':True,'artifact_sha256':hashlib.sha256(raw).hexdigest(),'source_artifact':str(f),'raw_max_rel_error_not_interpreted_as_fractional_error':q.get('max_rel_error')})
  records[model]=rows
 out['configurations'][r['configuration_id']]={'raw_result':str(run),'raw_result_sha256':hashlib.sha256(run.read_bytes()).hexdigest(),'archived_config_sha256':hashlib.sha256((run.parent/'config.json').read_bytes()).hexdigest(),'models':records}
d=R/'.porting-state/porting_logs/meanshift-julia-jacc';(d/'accuracy_checks.json').write_text(json.dumps(out,indent=2)+'\n');(P/'metric_definition_erratum.json').write_text(json.dumps({'classification_only':True,'raw_execution_unchanged':True,**out},indent=2)+'\n');print('PASS:40savedartifacts exactinput/alloutputbytes, trueCPUabsolute/relativeerrors/sourceclustercondition; metriclabelerratum preserved.')
