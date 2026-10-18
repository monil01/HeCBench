#!/usr/bin/env python3
"""CPU-only every saved raw key/pair artifact revalidation, NumPy2.5.4."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
P=Path(__file__).resolve().parent;N=1048576;results={};rows=[]
for mode in ['float_keys','float_pairs','uint_keys','uint_pairs']:
 log=P.parents[1]/'.porting-state/portable_julia'/f'radixsort2_jacc_{mode}_attempt01.log'
 rp=Path(json.loads(log.read_text().splitlines()[-1])['result'])
 r=json.loads(rp.read_text());assert r['matched_performance_valid'];fp=mode.startswith('float');only=mode.endswith('keys');dtype='<f4'if fp else'<u4';tag='f32'if fp else'u32';input=np.fromfile(P/f'study_input_{tag}.bin',dtype=dtype);expected=np.fromfile(P/f'study_expected_keys_{tag}.bin',dtype=dtype);ids=np.arange(N,dtype='u4');prefix=np.r_[input.astype('f8'),([]if only else ids.astype('f8')),N,int(only),int(fp),100,32];stable_ids=None if only else np.fromfile(P/('study_expected_values_f32_u32.bin'if fp else'study_expected_values_u32.bin'),dtype='<u4')
 for model,m in r['models'].items():
  for t in m['trials']:
   path=Path(t['output_artifact']['saved_path']);a=np.fromfile(path,dtype='<f8');assert np.array_equal(a[:len(prefix)],prefix) and np.isfinite(a).all();keys=a[len(prefix):len(prefix)+N];assert np.array_equal(keys,expected) and np.all(keys[:-1]<=keys[1:]);stable=None
   if not only:
    vals=a[-N:];assert np.array_equal(vals,vals.astype('u4'));vals=vals.astype('u4');assert np.array_equal(np.sort(vals),ids) and np.array_equal(keys,input[vals]);stable=bool(np.array_equal(vals,stable_ids));assert stable
   rows.append({'mode':mode,'model':model,'trial':t['trial'],'all_keys_exact':True,'raw_pair_association_and_permutation_exact':not only,'raw_pair_ties_match_stable_CPU_current_case':stable,'input_prefix_values':len(prefix),'checked_values':len(a),'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 results[mode]={'result':str(rp),'means_seconds':{m:x['mean_seconds']for m,x in r['models'].items()},'native_and_framework_raw_order_agree':True}
print(json.dumps({'benchmark':'radixsort2','numpy':np.__version__,'all80artifacts_pass':len(rows)==80,'results':results,'Thrust_stability_API_guarantee_claimed':False,'current_fixture_tie_order_observed_only':True,'artifacts':rows},indent=2))
