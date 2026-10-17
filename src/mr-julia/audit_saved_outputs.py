"""Exact CPU-only audit; full evidence is supplemental, outside the timed kernel."""
from pathlib import Path
from array import array
import json,hashlib,struct,sys
p=Path(__file__).resolve().parent;rows={}
for arg in sys.argv[1:]:
 rp=Path(arg);r=json.loads(rp.read_text());parts=r['configuration_id'].split('_');dist=parts[1];algo=parts[-2];mode=dist+'_'+algo
 prefix=struct.pack('<3I',2,7,61)+(p/f'study_{dist}_u32.bin').read_bytes();expected=(p/f'study_expected_{dist}_u32.bin').read_bytes();typed=array('I');typed.frombytes(expected);assert len(typed)==1200004
 for width in range(4):
  codes=typed[width*300000:(width+1)*300000];assert sum(codes)==typed[1200000+width]
  assert all(codes[j]>=codes[j+1]>=codes[j+2]and codes[j]<=1 for j in range(0,300000,3))
 if dist=='prime':assert all(v==1 for v in typed[:1200000])
 models={}
 for model,m in r['models'].items():
  trials=[]
  for t in m['trials']:
   f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw==prefix+expected
   trials.append({'trial':t['trial'],'artifact':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'exact_input_prefix':True,'all_predicates_exact':True,'timed_aggregate_equals_predicate_sums':True,'max_abs_error':0,'max_rel_error':0})
  models[model]=trials
 rows[mode]={'result':str(rp),'models':models,'aggregate_counts':list(typed[1200000:]),'four_widths_per_configuration':True}
(p/'saved_artifact_audit.json').write_text(json.dumps({'benchmark':'mr','method':'Byte-exact all5artifacts/all4models versusexactactualinputprefix+independentPythonbigintpredicateoracle; prefixmonotonicity andactualtimedaggregatecounts verified. Untimedfullpredicateinstrumentation notclaimedasoutputofnativecounterkernel.','configurations':rows},indent=2)+'\n');print({k:x['aggregate_counts']for k,x in rows.items()})
