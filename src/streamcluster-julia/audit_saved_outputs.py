"""CPU-only saved-evidence review; no benchmark or GPU execution."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
F=R/'.porting-state/portable_julia/runs/20261018T113358078526Z_streamcluster/result.json'
r=json.loads(F.read_text());assert r['matched_performance_valid']and all(r['variant_verified'].values())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fixture=np.fromfile(P/'input_f32.bin',dtype='<f4').reshape(512,8);expected_prefix=np.r_[fixture.ravel().view('<u4'),np.array([512,8,2,4,512,64,1,256,16,1],dtype='<u4')]
rows=[];bindings={};reference=None
for model,s in r['models'].items():
 for p,h in s['adapter_file_hashes'].items():assert sha(p)==h;bindings[p]=h
 for trial in s['trials']:
  path=Path(trial['output_artifact']['saved_path']);raw=np.fromfile(path,dtype='<u4');assert len(raw)==335722 and np.array_equal(raw[:4106],expected_prefix)
  if reference is None:reference=raw
  assert np.array_equal(raw,reference)
  q=4106;gain_count=0;work_count=0;maxdistance=0.;gain_inputs=[]
  while raw[q]==0x11111111:
   _,n,d,x,K=map(int,raw[q:q+5]);q+=5;assert(n,d)==(512,8)and 0<=x<n and K>=1
   coords=raw[q:q+n*d].view('<f4').reshape(d,n).T;q+=n*d
   points=raw[q:q+4*n].reshape(n,4);q+=4*n
   weight=points[:,0].copy().view('<f4');assign=points[:,1].astype(int);cost=points[:,2].copy().view('<f4');table=points[:,3]
   assert np.isfinite(coords).all()and np.all(weight==1)and np.isfinite(cost).all()and np.all(assign<n)
   assert collections.Counter(map(tuple,coords))==collections.Counter(map(tuple,fixture))
   for a in np.unique(assign):assert len(set(map(int,table[assign==a])))==1
   # Independent vectorized double-precision distances, no source function calls.
   dc=((coords.astype('f8')-coords[x].astype('f8'))**2).sum(axis=1)*weight
   sc=dc.astype('f4');maxdistance=max(maxdistance,float(np.max(np.abs(sc.astype('f8')-dc))))
   assert np.all(np.abs(sc.astype('f8')-dc)<=2e-6+2e-6*np.abs(dc))
   assigned=((coords.astype('f8')-coords[assign].astype('f8'))**2).sum(axis=1)*weight
   assert np.all(np.abs(cost-assigned)<=2e-5+2e-5*np.abs(assigned))
   assert tuple(map(int,raw[q:q+4]))==(0x22222222,n,K,x);z=raw[q+4:q+5].view('<f4')[0];q+=5
   state=raw[q:q+n*(K+2)].reshape(n,K+2);q+=n*(K+2);switch=state[:,0];work=state[:,1:].copy().view('<f4')
   active=sc<cost;assert np.array_equal(switch,np.where(active,49,0))
   expected=np.zeros((n,K+1),dtype='f4');slot=np.where(active,K,table).astype(int);expected[np.arange(n),slot]=np.where(active,sc-cost,cost-sc)
   assert np.array_equal(work,expected)and np.isfinite(work).all();work_count+=work.size
   unused=6*n-n*(K+1)+16;assert np.all(raw[q:q+unused]==0x4b123456);q+=unused
   assert np.all(raw[q:q+32]==0x4b123456);q+=32
   assert np.all(raw[q:q+400]==0x5b);q+=400
   gain_count+=1;gain_inputs.append(dict(candidate=x,centers=K,facility_cost=float(z)))
  assert tuple(map(int,raw[q:q+3]))==(0x33333333,512,8);q+=3
  before=raw[q:q+512*11].reshape(512,11);q+=512*11
  assignments=before[:,0].astype(int);weights=before[:,1].copy().view('f4');costs=before[:,2].copy().view('f4');coords=before[:,3:].copy().view('f4')
  assert np.all(assignments<512)and np.all(weights==1)
  assert tuple(map(int,raw[q:q+3]))==(0x44444444,2,8);q+=3
  centers=raw[q:q+2*13].reshape(2,13);q+=2*13;assert q==len(raw)
  ids=centers[:,0].astype(int);assert np.array_equal(ids,np.unique(assignments))and np.array_equal(centers[:,1],[0,1])and np.all(centers[:,2]==1)
  finalweight=centers[:,3].copy().view('f4');finalcoords=centers[:,5:].copy().view('f4');assert np.isfinite(finalcoords).all()and finalweight.sum()==512
  maxcentroid=0.
  for j,a in enumerate(ids):
   group=coords[assignments==a];assert len(group)==finalweight[j]
   err=float(np.max(np.abs(finalcoords[j].astype('f8')-group.astype('f8').mean(axis=0))));maxcentroid=max(maxcentroid,err);assert err<1e-6
  assert np.max(np.abs((finalcoords.astype('f8')*finalweight[:,None]).sum(axis=0)-fixture.astype('f8').sum(axis=0)))<1e-3
  assert gain_count==32 and work_count==49152
  rows.append(dict(model=model,trial=trial['trial'],path=str(path),sha256=sha(path),count=len(raw),gain_calls=gain_count,exact_work_values=work_count,final_centers=2,final_mass=float(finalweight.sum()),max_FP64_distance_error=maxdistance,max_centroid_abs_error=maxcentroid,full_bit_identity=True,exact_prefix_bytes=16424,gain_inputs=gain_inputs))
out=dict(result=str(F.relative_to(R)),result_sha256=sha(F),CPU_only=True,no_GPU_rerun=True,NumPy=np.__version__,scope='Full everygain semantic input/work/switch/guard capture; precentroid assignments; all finalcenterdefinedfields; independent FP64distance/centroid/mass/moment. Not global optimality/full multichunk/defaultscale validation.',rows=rows,executed_source_bindings=bindings)
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('streamcluster',len(rows),'captures',len(bindings),'sourcebindings',max(x['max_centroid_abs_error']for x in rows))
