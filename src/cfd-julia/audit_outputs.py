"""Full independent CFD artifact replay; Python standard library only."""
from pathlib import Path
from array import array
import argparse,hashlib,json,math,sys
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent
assert sys.byteorder=='little'
def load(q):
 x=array('d');size=q.stat().st_size;assert size%8==0
 with q.open('rb')as f:x.fromfile(f,size//8)
 return x
fixture=load(root/'fixture.bin');gold=load(root/'oracle.bin');fi=1;prefix=array('d',[2]);cases=[]
assert fixture[0]==2
for ci in range(2):
 n,steps,ident=map(int,fixture[fi:fi+3]);fi+=3;assert steps==2000 and ident==ci;chunks=[]
 for size in [n,4*n,12*n,5,3,3,3,3]:
  v=fixture[fi:fi+size];assert len(v)==size;fi+=size;chunks.append(v)
 prefix.extend([n,steps,ident])
 for ix,z in enumerate(chunks+[array('d',[0])*(5*n),array('d',[0])*(5*n),array('d',[0])*(5*n),array('d',[0])*n]):
  guard=-9001.0 if ix==1 else -9001.25;prefix.extend([guard]*32);prefix.extend(z);prefix.extend([guard]*32)
 cases.append((n,chunks))
assert fi==len(fixture);j=json.loads(a.result.read_text());rows=[];seen={}
for model,spec in j['models'].items():
 for t in spec.get('trials',[]):
  art=t['output_artifact'];path=Path(art['saved_path']);h=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert h==art['sha256'];assert path.stat().st_size==art['count']*8
  if h not in seen:
   x=load(path);assert x[:len(prefix)]==prefix;k=len(prefix);g=0;maxabs=0.;maxbudget=0.;minrho=float('inf');minp=float('inf')
   for ci,(n,chunks) in enumerate(cases):
    for group in range(3):
     sizes=[5*n,5*n,5*n,n] if group!=1 else [5*n]*20
     for ix,size in enumerate(sizes):
      assert all(z==-9001.25 for z in x[k:k+32]);v=x[k+32:k+32+size];assert all(z==-9001.25 for z in x[k+32+size:k+64+size]);k+=size+64
      ref=gold[g:g+size];g+=size;assert len(ref)==size
      for got,want in zip(v,ref):
       assert math.isfinite(got);e=abs(got-want);budget=2e-4+2e-3*abs(want);assert e<=budget;maxabs=max(maxabs,e);maxbudget=max(maxbudget,e/budget)
      if size==5*n and ((group==0 and ix==0)or group==1 or(group==2 and ix==0)):
       for i in range(n):
        rho=v[i];vsq=sum(v[i+d*n]**2 for d in (1,2,3))/rho;pressure=(1.3999999761581421-1)*(v[i+4*n]-.5*vsq);assert rho>0 and pressure>0;minrho=min(minrho,rho);minp=min(minp,pressure)
        if ci==0:assert all(abs(v[i+q*n]-chunks[3][q])<=3e-6 for q in range(5))
        elif i>=97046:assert all(v[i+q*n]==v[97045+q*n]for q in range(5))
    for ix,z in enumerate(chunks):
     guard=-9001.0 if ix==1 else -9001.25;assert all(q==guard for q in x[k:k+32]);assert x[k+32:k+32+len(z)]==z;assert all(q==guard for q in x[k+32+len(z):k+64+len(z)]);k+=len(z)+64
   assert k==len(x)and g==len(gold);seen[h]={'full_actual_input_words':len(prefix),'independent_values':g,'max_abs_error':maxabs,'max_fraction_of_predeclared_budget':maxbudget,'min_density':minrho,'min_pressure':minp}
  rows.append({'model':model,'trial':t['trial'],'artifact_sha256':h,'full_saved_artifact_sha256_rechecked':True,'verdict':'pass','mathematical_check_reused_for_identical_hash':any(r['artifact_sha256']==h for r in rows)})
report={'run_id':j['run_id'],'rows':rows,'independent_distinct_complete_artifact_checks':seen,'dependencies':'Pythonstdlib only','scope':'Completeprefix exact; allguards; fullF64Euler2000stepreference/init/checkpoints/finalworking; positivity/FF/padmirror atcapturedstates. No allGPUintermediatestage claim.'};text=json.dumps(report,indent=2)+'\n'
if a.output:a.output.write_text(text)
else:print(text)
