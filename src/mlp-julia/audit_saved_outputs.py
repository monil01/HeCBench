"""CPU-only recheck of every retained MLP artifact; does not retime kernels."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;rows={};fixture=np.fromfile(p/'study_input_f32.bin',dtype='<f4').astype('<f8');prefix=fixture.tobytes()
for result in sys.argv[1:]:
 rp=Path(result);r=json.loads(rp.read_text());mode=r['configuration_id'].split('_')[-4]+'_'+r['configuration_id'].split('_')[-3]
 # The explicit suffix is b0_r0_fp32_v1.
 assert mode in ['b0_r0','b1_r0','b0_r1','b1_r1']
 cpu=np.fromfile(p/f'study_expected_{mode}_f64.bin',dtype='<f8');models={}
 for model,m in r['models'].items():
  trials=[]
  for t in m['trials']:
   f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();a=np.frombuffer(raw,dtype='<f8');assert len(a)==902410 and raw[:len(prefix)]==prefix;got=a[len(fixture):];diff=np.abs(got-cpu);assert np.isfinite(got).all() and diff.max()<=.01
   if 'r1'in mode:assert (got>=0).all()
   start=0;layers=[]
   for n in [65536,65536,1280]:
    layers.append({'values':n,'max_abs_error':float(diff[start:start+n].max()),'max_relative_error_floor_0_01':float((diff[start:start+n]/np.maximum(np.abs(cpu[start:start+n]),.01)).max())});start+=n
   trials.append({'trial':t['trial'],'artifact':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'exact_input_prefix':True,'finite':True,'max_abs_error':float(diff.max()),'max_relative_error_floor_0_01':float((diff/np.maximum(np.abs(cpu),.01)).max()),'layer_errors':layers})
  models[model]=trials
 rows[mode]={'result':str(rp),'configuration_id':r['configuration_id'],'models':models}
out={'benchmark':'mlp','method':'IndependentFloat64densechain, all3layers/all5trials/4models, exact serialized actualinputprefix, finite andReLUnonnegative; no GPU execution or timing changes','numpy_version':np.__version__,'configurations':rows}
(p/'saved_artifact_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:{m:max(t['max_abs_error']for t in ts)for m,ts in x['models'].items()}for k,x in rows.items()})
