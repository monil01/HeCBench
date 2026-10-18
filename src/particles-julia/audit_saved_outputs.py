#!/usr/bin/env python3
"""CPU-only independent revalidation of every primary saved state. NumPy2.5.4."""
from pathlib import Path
import sys,json,hashlib,struct
import numpy as np
P=Path(__file__).resolve().parent;result=Path(sys.argv[1]);x=json.loads(result.read_text());N=16384;C=262144;K=131100
pos=np.fromfile(P/'study_input_pos_f32.bin',dtype='<f4');vel=np.fromfile(P/'study_input_vel_f32.bin',dtype='<f4');params=struct.unpack('<9f4I6f2I5f',(P/'study_params.bin').read_bytes());prefix=np.r_[pos.astype('f8'),vel.astype('f8'),params,.5,1];assert len(prefix)==K
cpu=np.fromfile(P/'study_expected_step1_f64.bin',dtype='<f8').reshape(3,N,4);rows=[]
for model,m in x['models'].items():
 for t in m['trials']:
  a=np.fromfile(t['output_artifact']['saved_path'],dtype='<f8');assert len(a)==950300 and np.array_equal(a[:K],prefix);v=a[K:];po=v[:4*N].reshape(N,4);ve=v[4*N:8*N].reshape(N,4);off=8*N;keys=v[off:off+N].astype('u4');off+=N;idx=v[off:off+N].astype('u4');off+=N;cs=v[off:off+C].astype('u4');off+=C;ce=v[off:off+C].astype('u4');off+=C;rp=v[off:off+4*N].reshape(N,4);off+=4*N;rv=v[off:].reshape(N,4)
  assert np.array_equal(np.sort(idx),np.arange(N));assert np.all(keys[:-1]>=keys[1:]);fp=po.astype('f4');origin=np.array(params[13:16],dtype='f4');cell=np.array(params[16:19],dtype='f4');gp=np.floor((fp[:,:3]-origin)/cell).astype('i4')&63;hashes=((gp[:,2]*64+gp[:,1])*64+gp[:,0]).astype('u4');assert np.array_equal(keys,hashes[idx]);starts=np.full(C,0xffffffff,dtype='u4');ends=starts.copy();first=np.r_[True,keys[1:]!=keys[:-1]];last=np.r_[keys[:-1]!=keys[1:],True];starts[keys[first]]=np.flatnonzero(first);ends[keys[last]]=np.flatnonzero(last)+1;assert np.array_equal(cs,starts) and np.array_equal(ce,ends)
  assert np.array_equal(rp,po[idx]);assert np.all(po[:,3]==1) and np.all(ve[:,3]==0);assert np.isfinite(a).all()
  d=np.r_[(po-cpu[0]).ravel(),(ve-cpu[1]).ravel(),(rp-cpu[0,idx]).ravel(),(rv-cpu[2,idx]).ravel()];ma=float(np.max(np.abs(d)));assert ma<=1e-5
  rows.append({'model':model,'trial':t['trial'],'full_state_cpu_max_abs':ma,'exact_prefix_values':K,'exact_hash_sort_permutation_cell_properties':True,'artifact_sha256':hashlib.sha256(Path(t['output_artifact']['saved_path']).read_bytes()).hexdigest()})
print(json.dumps({'benchmark':'particles','numpy':np.__version__,'primary_scope':'one full physical step only','all_pass':True,'artifacts':rows,'strict_eight_step_not_promoted':True},indent=2))
