from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
R=Path('/home/imo/HeCBench_julia');B=R/'src/aop-julia';W=R/'.porting-state/portable_julia/aop_reference_review'
# Actual original C++ RNG types/default seed/loop, generated once outside all timed regions.
(B/'source_rng_witness.cpp').write_text('#include <random>\n#include <fstream>\nint main(){std::default_random_engine rng;std::normal_distribution<double> dist(0.,1.);std::ofstream f("source_normals.bin",std::ios::binary);for(int i=0;i<100*32768;i++){double x=dist(rng);f.write((char*)&x,8);}}\n')
subprocess.run(['g++','-O3','-std=c++17',str(B/'source_rng_witness.cpp'),'-o',str(W/'rng')],check=True);subprocess.run([str(W/'rng')],cwd=W,check=True);normal=np.fromfile(W/'source_normals.bin',dtype='<f8')
cases=[dict(n=256,steps=3,K=1.,S0=.5,r=.03,sigma=.7,put=True),dict(n=512,steps=8,K=1.,S0=1.,r=.03,sigma=.7,put=False),dict(n=256,steps=3,K=1.,S0=4.,r=0.,sigma=0.,put=True),dict(n=256,steps=3,K=1.,S0=.5,r=0.,sigma=0.,put=True),dict(n=1024,steps=12,K=4.,S0=3.6,r=.06,sigma=.2,put=True),dict(n=32768,steps=100,K=4.,S0=3.6,r=.06,sigma=.2,put=True),dict(n=256,steps=3,K=1.,S0=1.,r=.005,sigma=.1,put=True)]
inputs=[];oracles=[];size=0;osize=0;descs=[]
for ci,c in enumerate(cases):
 n=c['n'];steps=c['steps'];samples=normal[:n*steps].copy().reshape(steps,n)
 if ci==6:samples[:,:]=-4.;samples[:,3:]=4.
 S=np.full(n,c['S0']);path=[];dt=1/steps
 for t in range(steps):
  S*=np.exp((c['r']-.5*c['sigma']**2)*dt+c['sigma']*np.sqrt(dt)*samples[t]);path.append(S.copy())
 paths=np.array(path);pay=lambda s:np.maximum(c['K']-s,0.)if c['put']else np.maximum(s-c['K'],0.);cash=pay(paths[-1]);discount=np.exp(-c['r']*dt);ranks=[];margins=[]
 for t in range(steps-2,-1,-1):
  x=paths[t];inside=x<c['K']if c['put']else x>c['K'];m=int(inside.sum());beta=np.zeros(3)
  if m>=4:
   A=np.array([np.ones(m),x[inside],x[inside]**2]).T
   # Independent stable SVD least-squares. Native source forms R^T R and cuts eigenvalues1e-12.
   U,s,Vh=np.linalg.svd(A,full_matrices=False);keep=s*s>=1e-12;beta=Vh.T@np.where(keep,(U.T@cash[inside])/s,0.);ranks.append(dict(t=t,m=m,singular_values=s.tolist(),rank=int(keep.sum()),condition=float(s[0]/s[-1])if s[-1]>0 else None))
  p=pay(x);estimated=(beta[0]+beta[1]*x+beta[2]*x*x)*discount;take=(p>1e-8)&(p>estimated)if m>=4 else np.zeros(n,dtype=bool);margins.append(dict(t=t,in_the_money=m,exercise_count=int(take.sum()),min_positive_decision_margin=float(np.abs(p[inside]-estimated[inside]).min())if m>=4 else None));cash=np.where(take,p,discount*cash)
 paths[-1]=cash;price=discount*cash.sum()/n;c.update(case_id=ci,dt=dt,io=size,oo=osize,source_full_default=ci==5);flat=samples.ravel();inputs.append(flat);oracles.append(paths.ravel());size+=len(flat);osize+=paths.size;descs.append(dict(case=ci,final_independent_price=price,regression_rank_controls=ranks,exercise_decisions=margins,limitation='Bermudan finite-path estimator at dt..T, not t0 immediateexercise nor exact American price; source default QR normalmatrix1e-12 cutoff retained and independently audited.'))
np.concatenate(inputs).astype('<f8').tofile(B/'input_f64.bin');np.concatenate(oracles).astype('<f8').tofile(B/'oracle_paths_f64.bin')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(B/'cases.json').write_text(json.dumps(dict(cases=cases,independent=descs,CPU_generator=dict(python=sys.version,numpy=np.__version__),input=dict(bytes=(B/'input_f64.bin').stat().st_size,sha256=sha(B/'input_f64.bin')),oracle=dict(bytes=(B/'oracle_paths_f64.bin').stat().st_size,sha256=sha(B/'oracle_paths_f64.bin'))),indent=2)+'\n')
fields=['n','steps','K','S0','r','sigma','put','dt','io','oo']
(B/'study_cases.h').write_text('struct Case{int n,steps;double K,S0,r,sigma;bool put;double dt;int io,oo;};\nconst Case cases[]={\n'+',\n'.join('{'+','.join(str(c[k]).lower()for k in fields)+'}'for c in cases)+'\n};\n')
(B/'study_cases.jl').write_text('const CASES=[\n'+',\n'.join('(;'+','.join(k+'='+str(c[k]).lower()for k in fields)+')'for c in cases)+'\n]\n')
print(size,osize,[x['final_independent_price']for x in descs])
