"""Independent replay of complete CFD artifacts, with exact actual-input checks."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent
fixture=np.fromfile(root/'fixture.bin',dtype='<f8');gold=np.fromfile(root/'oracle.bin',dtype='<f8');fi=1;prefix=[np.array([2.])];cases=[]
assert fixture[0]==2
for ci in range(2):
 n,steps,ident=map(int,fixture[fi:fi+3]);fi+=3;assert steps==2000 and ident==ci
 chunks=[]
 for size in [n,4*n,12*n,5,3,3,3,3]:
  v=fixture[fi:fi+size];assert len(v)==size;fi+=size;chunks.append(v)
 pref=[np.array([n,steps,ident],dtype=np.float64)]
 for ix,z in enumerate(chunks+[np.zeros(5*n),np.zeros(5*n),np.zeros(5*n),np.zeros(n)]):
  guard=-9001.0 if ix==1 else -9001.25
  pref.extend([np.full(32,guard),z,np.full(32,guard)])
 prefix.extend(pref);cases.append((n,chunks))
assert fi==len(fixture);expectedprefix=np.concatenate(prefix)
j=json.loads(a.result.read_text());rows=[];seen={}
for model,spec in j['models'].items():
 for t in spec.get('trials',[]):
  art=t['output_artifact'];path=Path(art['saved_path']);h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==art['sha256'];assert path.stat().st_size==art['count']*8
  if h not in seen:
   x=np.fromfile(path,dtype='<f8');assert np.array_equal(x[:len(expectedprefix)],expectedprefix);k=len(expectedprefix);g=0;maxabs=0.;maxbudget=0.;minrho=float('inf');minp=float('inf')
   for ci,(n,chunks) in enumerate(cases):
    for group in range(3):
     sizes=[5*n,5*n,5*n,n] if group!=1 else [5*n]*20
     for ix,size in enumerate(sizes):
      assert np.all(x[k:k+32]==-9001.25);v=x[k+32:k+32+size];assert np.all(x[k+32+size:k+64+size]==-9001.25);k+=size+64
      ref=gold[g:g+size];g+=size;assert len(ref)==size and np.all(np.isfinite(v));e=np.abs(v-ref);budget=2e-4+2e-3*np.abs(ref);assert np.all(e<=budget);maxabs=max(maxabs,float(e.max(initial=0)));maxbudget=max(maxbudget,float((e/budget).max(initial=0)))
      if size==5*n and ((group==0 and ix==0)or group==1 or(group==2 and ix==0)):
       u=v.reshape(5,n);rho=u[0];pressure=(float(np.float32(1.4))-1)*(u[4]-.5*np.sum(u[1:4]*u[1:4],axis=0)/rho);assert np.all(rho>0)and np.all(pressure>0);minrho=min(minrho,float(rho.min()));minp=min(minp,float(pressure.min()))
       if ci==0:assert np.max(np.abs(u-np.array(chunks[3])[:,None]))<=3e-6
       else:assert np.array_equal(u[:,97046:],np.broadcast_to(u[:,97045:97046],(5,n-97046)))
    for ix,z in enumerate(chunks):
     guard=-9001.0 if ix==1 else -9001.25;assert np.all(x[k:k+32]==guard);assert np.array_equal(x[k+32:k+32+len(z)],z);assert np.all(x[k+32+len(z):k+64+len(z)]==guard);k+=len(z)+64
   assert k==len(x)and g==len(gold)
   seen[h]={'full_actual_input_words':len(expectedprefix),'independent_values':g,'max_abs_error':maxabs,'max_fraction_of_predeclared_budget':maxbudget,'min_density':minrho,'min_pressure':minp}
  rows.append({'model':model,'trial':t['trial'],'artifact_sha256':h,'full_saved_artifact_sha256_rechecked':True,'verdict':'pass','mathematical_check_reused_for_identical_hash':sum(r['artifact_sha256']==h for r in rows)>0})
report={'run_id':j['run_id'],'rows':rows,'independent_distinct_complete_artifact_checks':seen,'numpy_version':np.__version__,'scope':'Completeprefix exact; allguards; fullF64Euler2000stepreference/init/checkpoints/finalworking; positivity/FF/padmirror atcapturedstates. No allGPUintermediatestage claim.'}
text=json.dumps(report,indent=2)+'\n'
if a.output:a.output.write_text(text)
else:print(text)
