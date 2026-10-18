from pathlib import Path
import math,struct,json
p=Path(__file__).resolve().parent;F=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];MASK=(1<<32)-1;N=256;PATHS=512
# Independent integer state transition in Python's unbounded integers, explicit32bit masks.
def uniform(z):
 params=[(13,19,12,4294967294),(2,25,4,4294967288),(3,11,17,4294967280)]
 zz=[]
 for s,(a,b,c,m) in zip(z[:3],params):zz.append(((((s&m)<<c)&MASK)^((((s<<a)&MASK)^s)>>b))&MASK)
 zz.append((1664525*z[3]+1013904223)&MASK);return F(F(zz[0]^zz[1]^zz[2]^zz[3])*F(2**-32)),zz
seedplanes=[[((i*747796405+(k+1)*2891336453)&MASK)|16 for i in range(N)] for k in range(4)]
seeds=[s for arr in seedplanes for s in arr];prefix=[N,4,PATHS]+seeds;gold=[];cases=[];guard=-12345.;all_rng=[]
for ci,T in enumerate([15,2,15,0]):
 deterministic=ci==2;params=[[] for _ in range(7)]
 for i in range(N):
  vals=[0 if deterministic else (4+i%5)/64,0 if deterministic else ((i%7)-3)/512,0 if deterministic else 1/1024,0 if deterministic else 1/16,0 if deterministic else 1/4,1+(i%11)/32,0 if deterministic else (i%5-2)/2048]
  for a,v in zip(params,vals):a.append(F(v))
 prefix+=[T]+[x for a in params for x in a];means=[];variances=[];probe=[];umin=1.;umax=0.;maxvol=0.;maxprice=0.;maxres=0.
 for i in range(N):
  z=[a[i] for a in seedplanes];first=[];mean=0.;m2=0.
  for sim in range(PATHS):
   vol,eps,a0,a1,a2,s,mu=[a[i] for a in params];past=[];temporary=0.
   for t in range(T):
    past.append(s);vol=math.sqrt(a0+a1*vol*vol+a2*eps*eps);maxvol=max(maxvol,vol)
    if t%2==0:
     u1,z=uniform(z);u2,z=uniform(z);assert u1>0 and u1<=1 and 0<=u2<=1;umin=min(umin,u1,u2);umax=max(umax,u1,u2)
     if sim==0 and t==0:first=[u1,u2]
     rad=math.sqrt(-2*math.log(u1));angle=F(2*F(math.pi))*u2;r=rad*math.sin(angle);temporary=rad*math.cos(angle)
    else:r=temporary
    eps=max(-1.,min(1.,r*vol));s*=math.exp(mu+eps);maxprice=max(maxprice,s)
   res=sum(max(x-s,0.) for x in reversed(past));maxres=max(maxres,res);delta=res-mean;mean+=delta/(sim+1);m2+=delta*(res-mean)
  means.append(mean);variances.append(m2/(PATHS-1));probe+=z+(first if first else [0.,0.])
 for a in [means,variances]:gold+=[guard]*8+a+[guard]*8
 # Actual outputs of a separate RNG-only consistency probe: final4UInt32states+first2F32uniforms perlane.
 gold+=probe
 for ai,a in enumerate([seeds]+params):
  marker=1515870810 if ai==0 else guard;gold+=[marker]*8+a+[marker]*8
 cases.append(dict(case=ci,cycles=T,deterministic_zero_payoff=deterministic or T==0,oracle='Independent Float64 recurrence/payoff/Welford with exactUInt32RNGandF32uniforminputs; transcendental/arithmetic rounding differsGPUF32 withinpredeclaredbudget',min_observed_uniform=umin if T else None,max_observed_uniform=umax if T else None,max_volatility=maxvol,max_price=maxprice,max_payoff=maxres,draws_per_lane=PATHS*2*((T+1)//2),mean_range=[min(means),max(means)],variance_range=[min(variances),max(variances)]))
for name,a in [('fixture.bin',prefix),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(N=N,paths_per_parameter=PATHS,cases=cases,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),atol=.002,rtol=.002,mathscope='Actualsourcefunctional sum_t max(S_t-S_terminal,0), notstandard max-path lookback/discountedoptionprice. Namedsmalloneblock256lanecompactseedstride256insteadsource4096blocks/1mstride, originalGPUbodyunchanged underexplicitmacroconfig.',uniform_scope='ExactUInt32transition/actualF32uniformrounding. Min/maxuniformCPUdomaincheck andfinalstates/first2uniformGPUprobeoutsideROI, notstatisticaldistributionquality orcompleteallgeneratedrandomarraycapture.',seed_guard_word=1515870810,premeasurement_guard_representation='Changed only UInt32 seedguards from unavailable-negative numeric sentinel to0x5a5a5a5a; generator source updated, all oracle means/variances/RNGstates unchanged beforefirstGPUmeasurement'),indent=2)+'\n')
print(json.dumps(cases))
