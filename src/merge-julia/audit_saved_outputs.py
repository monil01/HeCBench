#!/usr/bin/env python3
"""Exact bytes/type/path/multiset evidence from immutable artifacts; no GPU calls."""
from pathlib import Path
import array,json,hashlib,sys,struct
P=Path(__file__).resolve().parent;R=P.parent.parent;formats={'uint32':'I','uint64':'Q','float32':'f','float64':'d'};out={'configurations':{},'claim':'Positive finite/unsigned scalar merge outputs and exactpartition coordinates; no payload stability/negativefloat/NaN claim.'}
for arg in sys.argv[1:]:
 run=Path(arg);run=run if run.is_absolute()else R/run;r=json.loads(run.read_text());assert r['matched_performance_valid'];mode=r['configuration_id'].split('_')[-3];assert mode in formats;fmt=formats[mode];input_bytes=(P/f'study_input_{mode}.bin').read_bytes();merged=(P/f'study_expected_{mode}.bin').read_bytes();path=array.array('I');path.frombytes((P/f'study_paths_{mode}.bin').read_bytes());typedpath=array.array(fmt,path).tobytes();expected=input_bytes+merged+typedpath;records={}
 for model,x in r['models'].items():
  rows=[]
  for q in x['trials']:
   f=Path(q['output_artifact']['saved_path']);raw=f.read_bytes();assert raw==expected;values=array.array(fmt);values.frombytes(raw);assert len(values)==402274;data=values[202048:402048];assert all(data[i]<=data[i+1]for i in range(199999))
   rows.append({'trial':q['trial'],'input_prefix_bytes_exact':True,'full200000merged_values_exact':True,'all226independent_stable_merge_path_coordinates_exact':True,'output_multiset_and_order_preserved':True,'actual_storage_dtype':mode,'UInt64_bits_preserved_without_Float64_conversion':mode=='uint64','artifact_sha256':hashlib.sha256(raw).hexdigest(),'saved_artifact':str(f)})
  records[model]=rows
 out['configurations'][r['configuration_id']]={'raw_result':str(run),'raw_result_sha256':hashlib.sha256(run.read_bytes()).hexdigest(),'archived_config_sha256':hashlib.sha256((run.parent/'config.json').read_bytes()).hexdigest(),'models':records}
d=R/'.porting-state/porting_logs/merge-julia-jacc';(d/'exact_copy_checks.json').write_text(json.dumps(out,indent=2)+'\n');(P/'artifact_provenance.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS:80savedartifacts exactactualinputprefix/fullmergedcopies/paths; UInt64LSBs preserved, no mathematicalcopytolerance.')
