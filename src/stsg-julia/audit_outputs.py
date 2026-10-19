"""CPU-only artifact rehash/full-state audit with independent constituent mathematics.
Usage: python3 src/stsg-julia/audit_outputs.py RESULT.json
Full reconstruction oracle is source-derived; independent constituent checks do
not certify remote-sensing accuracy or an independent full reconstruction.
"""
from pathlib import Path
import array,hashlib,json,math,struct,sys
P=Path(__file__).resolve().parent;R=P.parents[1]
def load(p):
 a=array.array('d');a.frombytes(Path(p).read_bytes());assert sys.byteorder=='little';return a
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
f=load(P/'study_fixture.bin');oracle=load(P/'study_oracle.bin');manifest=json.loads((P/'fixture_manifest.json').read_text());cases=[];prefix=[f[0]];off=1
for c in manifest['cases']:
 head=list(f[off:off+8]);off+=8;X,Y,B,Z,w=map(int,head[:5]);N=X*Y;Q=N*B*Z;W=(2*w+1)**2;raw=list(f[off:off+Q]);off+=Q;qa=list(f[off:off+Q]);off+=Q
 inputs=[raw,qa]+[[0.]*n for n in [Q,Q,N*B,N*W*4,N*W*3,Q,N*B,N*B,N*W*3]]
 guarded=[([173. if k==1 else -37.]*32+v+[173. if k==1 else -37.]*32)for k,v in enumerate(inputs)]
 prefix+=head
 for v in guarded:prefix+=v
 cases.append((head,raw,qa,guarded))
assert off==len(f)
def audit(a):
 assert list(a[:len(prefix)])==prefix;pos=len(prefix);values=a[pos:pos+len(oracle)];pos+=len(oracle)
 assert all(math.isfinite(x) and abs(x-y)<=2e-5+2e-5*abs(y)for x,y in zip(values,oracle))
 it=0;err=dict(conversion=0.,reference_ensemble_mean=0.,Pearson_reference=0.,ratio_slope=0.,ratio_intercept=0.,ratio_Pearson=0.);counts={k:0 for k in err};mincos=1.;lowzero=0
 def take(n):
  nonlocal it
  v=values[it:it+n];it+=n;return v
 def check(key,x,y,budget):
  e=abs(x-y);err[key]=max(err[key],e);counts[key]+=1;assert math.isfinite(x) and e<=budget,(key,x,y,budget)
 for head,raw,qa,guarded in cases:
  X,Y,B,Z,w=map(int,head[:5]);N=X*Y;Q=N*B*Z;W=(2*w+1)**2
  nd1,q1=take(Q),take(Q);nd,q=take(Q),take(Q);ref=take(N*B);take(N*W*3);take(N*B);res=take(N*W*4)
  for k in range(Q):check('conversion',nd1[k],f32(f32(raw[k])/10000.),2e-7);assert q1[k]==qa[k]
  # These declared fixtures select all years: independently check each pair's
  # good/marginal-mask cosine, then compute the full arithmetic ensemble mean.
  for cell in range(N):
   for y in range(Z):
    for z in range(y+1,Z):
     pairs=[(nd1[cell+k*N+y*N*B],nd1[cell+k*N+z*N*B])for k in range(B)if q1[cell+k*N+y*N*B]in(0.,1.)and q1[cell+k*N+z*N*B]in(0.,1.)]
     cos=sum(x*y for x,y in pairs)/math.sqrt(sum(x*x for x,y in pairs)*sum(y*y for x,y in pairs));mincos=min(mincos,cos);assert cos>head[5]
   for k in range(B):
    v=[nd1[cell+k*N+y*N*B]for y in range(Z)if q1[cell+k*N+y*N*B]in(0.,1.)]
    assert v;check('reference_ensemble_mean',ref[cell+k*N],sum(v)/len(v),2e-7)
  for j in range(Y):
   for i in range(X):
    center=i+j*X
    for dy in range(-w,w+1):
     for dx in range(-w,w+1):
      if not(0<=i+dx<X and 0<=j+dy<Y):continue
      neighbor=i+dx+(j+dy)*X;idx=center*W*4+dx+w+(dy+w)*(2*w+1)
      xx=[ref[neighbor+k*N]for k in range(B)];yy=[ref[center+k*N]for k in range(B)];xm=sum(xx)/B;ym=sum(yy)/B
      cov=sum((x-xm)*(y-ym)for x,y in zip(xx,yy));vx=sum((x-xm)**2 for x in xx);vy=sum((y-ym)**2 for y in yy)
      check('Pearson_reference',res[idx],cov/math.sqrt(vx*vy),2e-7)
      pairs=[(f32(nd[neighbor+k*N+y*N*B]/ref[neighbor+k*N]),f32(nd[center+k*N+y*N*B]/ref[center+k*N]))for k in range(B)for y in range(Z)if q[neighbor+k*N+y*N*B]in(0.,1.)and q[center+k*N+y*N*B]in(0.,1.)]
      if len(pairs)>=30:
       xm=sum(x for x,y in pairs)/len(pairs);ym=sum(y for x,y in pairs)/len(pairs);cv=sum((x-xm)*(y-ym)for x,y in pairs);vx=sum((x-xm)**2 for x,y in pairs);vy=sum((y-ym)**2 for x,y in pairs);slope=cv/vx
       check('ratio_slope',res[idx+W],slope,2e-5+2e-5*abs(slope));check('ratio_intercept',res[idx+W*2],ym-slope*xm,2e-5+2e-5*abs(ym-slope*xm));check('ratio_Pearson',res[idx+W*3],cv/math.sqrt(vx*vy),2e-7)
  final=[take(n)for n in [Q,Q,N*B,N*W*4,N*W*3,Q,N*B,N*B,N*W*3]]
  if X==7:assert all(x==0 for x in final[5]);lowzero+=len(final[5])
  for initial in guarded:
   assert list(a[pos:pos+32])==initial[:32];pos+=32;assert list(a[pos:pos+32])==initial[-32:];pos+=32
  for initial in guarded[:2]:assert list(a[pos:pos+len(initial)])==initial;pos+=len(initial)
 assert it==len(oracle) and pos==len(a)
 return {'all_pass':True,'max_full_source_oracle_abs':max(abs(x-y)for x,y in zip(values,oracle)),'independent_max_abs':err,'independent_checked_values':counts,'minimum_all_selected_year_pair_cosine':mincos,'manufactured_low_NDVI_zero_values':lowzero,'scope_limit':'Independent full constituent conversion/ensemble/Pearson/OLS checks plus lowNDVI/spline controls; full reconstruction oracle is source-derived, not independent physical accuracy.'}
if __name__=='__main__':
 path=Path(sys.argv[1]);path=path if path.is_absolute()else R/path;d=json.loads(path.read_text());cache={};rows=[]
 for m,rec in d['models'].items():
  if not rec['trials']:continue
  assert len(rec['trials'])==5;assert math.isclose(rec['mean_seconds'],sum(t['time_seconds']for t in rec['trials'][1:])/4,rel_tol=1e-12)
  for t in rec['trials']:
   art=t['output_artifact'];p=Path(art['saved_path']);digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==art['sha256']
   if digest not in cache:cache[digest]=audit(load(p))
   rows.append({'model':m,'trial':t['trial'],'sha256':digest,'audit':cache[digest]})
 out={'run_id':d['run_id'],'all_pass':True,'checked_trials':len(rows),'artifact_hash_classes':len(cache),'checks':rows};(P/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k!='checks'});print(json.dumps(next(iter(cache.values())),indent=2))
