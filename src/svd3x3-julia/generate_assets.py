#!/usr/bin/env python3
"""Regenerate measured inputs and independent sigma assets without GPU captures."""
from pathlib import Path
import numpy as np,argparse,json,hashlib
P=Path(__file__).resolve().parent;a=argparse.ArgumentParser();a.add_argument('--output-dir',type=Path,default=P);opts=a.parse_args();out=opts.output_dir;out.mkdir(parents=True,exist_ok=True)
m=json.loads((P/'original_asset_manifest.json').read_text());path=P/'original_assets/Dataset_1M.txt';assert hashlib.sha256(path.read_bytes()).hexdigest()==m['files'][0]['sha256'];text=path.read_text();header,body=text.split('\n',1);assert int(header)==1048576;A=np.fromstring(body,sep=' ',dtype='<f4');assert len(A)==9*1048576;A=A.reshape(9,1048576)
controls=[np.zeros((3,3)),np.diag([3.,2.,1.]),np.diag([-3.,2.,-1.]),np.diag([1.,1.,1.]),np.diag([5.,0.,0.]),np.diag([4.,2.,0.]),np.ones((3,3)),np.array([[0,3,0],[0,0,-2],[1,0,0.]])]
for j in range(23):
 t=(j+1)*.137;c=np.cos(t);s=np.sin(t);Q=np.array([[c,-s,0],[s,c,0],[0,0,1.]]);v=(j+1)*.179;c=np.cos(v);s=np.sin(v);V=np.array([[1,0,0],[0,c,-s],[0,s,c]]);controls.append(Q@np.diag([2.5,1.+j*.01,.25])@V.T)
plan=json.loads((P/'study_plan.json').read_text())
for case in plan['cases']:
 k=case['index'];n=case['n']
 if k==0:B=np.zeros((9,n),dtype='<f4')
 elif k==1:B=np.asarray(controls,dtype='<f4').reshape(31,9).T.copy()
 else:
  ix=np.linspace(0,1048575,n,dtype=np.int64);ix[0]=373132;ix[1]=784340;B=np.asarray(A[:,ix],dtype='<f4')
 sigma=np.linalg.svd(B.T.astype('f8').reshape(n,3,3),compute_uv=False)
 for name,b,expected in [(f'case{k}_input_f32.bin',np.ascontiguousarray(B).tobytes(),case['input_sha256']),(f'case{k}_sigma_f64.bin',np.ascontiguousarray(sigma).tobytes(),case['independent_sigma_sha256'])]:
  assert hashlib.sha256(b).hexdigest()==expected,name;(out/name).write_bytes(b)
print('12 measured asset hashes reproduced from original text and CPU LAPACK')
