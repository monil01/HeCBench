#!/usr/bin/env python3
"""Independent full captured-input sum audit; no GPU execution or tolerance change."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]).resolve();d=json.loads(p.read_text());rows=[]
for model,m in d['models'].items():
 for trial in m.get('trials',[]):
  art=trial['output_artifact'];path=Path(art.get('saved_path',art['path']));path=path if path.is_absolute() else p.parent/path
  a=np.memmap(path,dtype='<f4',mode='r');assert a.size==92274872
  pre=0;post=46137436;count=0
  for case,n in enumerate((17,1048576,10485760)):
   width=n+2;b=a[pre:pre+4*width].reshape(2,2,width);o=a[post:post+4*width].reshape(2,2,width)
   assert np.all(b[:,:,0]==4096.25) and np.all(b[:,:,-1]==4096.25)
   assert np.all(o[:,:,0]==4096.25) and np.all(o[:,:,-1]==4096.25)
   assert np.array_equal(b[:,0,:],o[:,0,:]) and np.all(b[:,1,1:-1]==-777)
   expected=b[0,0,1:-1].astype(np.float64)+b[1,0,1:-1].astype(np.float64)
   for rank in range(2):assert np.array_equal(o[rank,1,1:-1].astype(np.float64),expected)
   if case==0:assert np.all(expected==0)
   if case==1:assert np.all(b[:,0,1:-1]==1) and np.all(expected==2)
   pre+=4*width;post+=4*width;count+=2*n
  assert pre==46137436 and post==92274872
  h=hashlib.sha256()
  with path.open('rb') as f:
   for part in iter(lambda:f.read(2**20),b''):h.update(part)
  assert not art.get('sha256') or h.hexdigest()==art['sha256']
  rows.append({'model':model,'trial':trial['trial'],'path':str(path),'sha256':h.hexdigest(),'exact_output_checks':count,'actual_input_initial_output_readonly_guards':'pass'})
print(json.dumps({'result':str(p),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pass':True,'captures':rows,'method':'Float64 direct rank-input sum from actual full saved prefix, complete two-rank output/readonly/guards; exact dyadic sums, independent Python implementation.'},indent=2))
