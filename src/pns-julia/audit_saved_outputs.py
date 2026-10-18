#!/usr/bin/env python3
"""CPU-only all saved full-state revalidation. No GPU calls."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;r=json.loads(Path(sys.argv[1]).read_text());Q=2097152;T=4;expected=np.fromfile(P/'study_expected_marks_u8.bin',dtype='u1').reshape(T,Q);conf=np.fromfile(P/'study_expected_conflicts_i32.bin',dtype='<i4').reshape(T,Q);rewards=np.fromfile(P/'study_expected_rewards_f64.bin',dtype='<f8').reshape(T,2);rows=[]
for model,m in r['models'].items():
 for trial in m['trials']:
  path=Path(trial['output_artifact']['saved_path']);a=np.fromfile(path,dtype='<f4');assert len(a)==16777228 and np.array_equal(a[:4],np.array([1024,312,4,5489],dtype='f4'));off=4;metrics=[]
  for b in range(T):
   marks=a[off:off+Q];off+=Q;cs=a[off:off+Q];off+=Q;reward=a[off:off+2];off+=2;assert np.array_equal(marks,expected[b]) and np.array_equal(cs,conf[b]) and np.array_equal(reward,rewards[b]);total=int(marks.astype('u8').sum());squares=int((marks.astype('u8')**2).sum());var=np.float32(np.float32(squares)/np.float32(Q))-np.float32(1);assert total==Q and var==reward[0] and marks.max()==reward[1];metrics.append({'trajectory':b,'seed':5489+b,'token_total':total,'squared_marking_sum':squares,'variance_reward':float(var),'maximum_mark':int(marks.max())})
  assert off==len(a) and np.isfinite(a).all();rows.append({'model':model,'trial':trial['trial'],'full_markings_conflict_reward_exact':True,'max_abs_error':0,'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'trajectory_statistics':metrics})
print(json.dumps({'benchmark':'pns','numpy':np.__version__,'all20artifacts_exact':len(rows)==20,'named_scope':'N1024/S312/4trajectories/one624attemptMTbatch/initialconflictszero','original_N22_long_ensemble_certified':False,'artifacts':rows},indent=2))
