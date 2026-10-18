"""Read-only complete-state replay plus independent direct Simpson convolution."""
from pathlib import Path
import json,struct,hashlib,sys,math
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];bundle=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bundle/'result.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f,n):
 b=Path(f).read_bytes();assert len(b)==8*n;return list(struct.unpack('<'+str(n)+'d',b))
pre=load(p/'fixture.bin',6023);gold=load(p/'oracle.bin',13164);desc=json.loads((p/'fixture_description.json').read_text());f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];L=f32(.7);mu=f32(.9);mu2=f32(.3);pos=1;inputs=[]
for c in desc['cases']:
 R,F,G,Q,FAI,repeat,mode=map(int,pre[pos:pos+7]);pos+=7;nf=R*F*G;ns=R*G
 source=pre[pos:pos+nf];pos+=nf;flux=pre[pos:pos+nf];pos+=nf;sig=pre[pos:pos+ns];pos+=ns;state=pre[pos:pos+G];pos+=G;inputs.append(dict(source=source,flux=flux,sig=sig,state=state))
assert pos==len(pre)
cache={}
def quadrature(q0,q1,q2,sigma,state):
 key=(q0,q1,q2,sigma,state)
 if key in cache:return cache[key]
 def evaluate(N):
  out=state*math.exp(-sigma*L);J=state*(-math.expm1(-sigma*L))/sigma;sout=sJ=0.
  for i in range(N+1):
   s=L*i/N;emission=q0+q1*mu*s+q2*mu2*s*s;w=1 if i in (0,N) else 4 if i%2 else 2
   sout+=w*emission*math.exp(-sigma*(L-s));sJ+=w*emission*(-math.expm1(-sigma*(L-s)))/sigma
  return out+sout*L/(3*N),J+sJ*L/(3*N)
 a=evaluate(128);b=evaluate(256);gap=max(abs(x-y) for x,y in zip(a,b));assert gap<1e-8
 cache[key]=(b,gap);return cache[key]
rows=[];hashes=set();fieldmax={};fieldmetrics={};maxnormalized=maxquadrature=maxgap=maxbalance=0.
for model,v in d['models'].items():
 assert v['complete'] and len(v['trials'])==5
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);a=load(f,19187);assert a[:6023]==pre and t['correct'];out=a[6023:];gp=0;cases=[];hashes.add(sha(f))
  for ci,c in enumerate(desc['cases']):
   R,F,G,Q,FAI,rep,mode=[c[k] for k in ['regions','fine_intervals','energy_groups','QSR','FAI','repeat','mode']];nf=R*F*G;ns=R*G;fields=[('fine_flux',nf),('state',G),*[(z,G) for z in ['q0','q1','q2','sigma','tau','sigma_squared','expVal','reuse','flux_integral','tally','t1','t2','t3','t4']],('source_echo',nf),('sigma_echo',ns),('initial_flux_echo',nf),('initial_state_echo',G),('ids_echo',2),('guards',112)];actual={}
   for name,n in fields:
    values=out[gp:gp+n];expected=gold[gp:gp+n];assert len(values)==n and all(math.isfinite(x) for x in values);errors=[abs(x-y) for x,y in zip(values,expected)];budgets=[2e-5+2e-5*abs(y) for y in expected];norm=max((e/b for e,b in zip(errors,budgets)),default=0);assert norm<=1
    if name.endswith('_echo') or name=='guards':assert values==expected
    maxnormalized=max(maxnormalized,norm);fieldmax[name]=max(fieldmax.get(name,0),max(errors,default=0));actual[name]=values;gp+=n
    fm=fieldmetrics.setdefault(name,dict(max_absolute_error=0.,max_relative_error_nonzero_reference=0.,max_zero_reference_absolute_error=0.,max_budget_fraction=0.,near_zero_reference_count_abs_below_1e_6=0))
    fm['max_absolute_error']=max(fm['max_absolute_error'],max(errors,default=0));fm['max_relative_error_nonzero_reference']=max(fm['max_relative_error_nonzero_reference'],max((e/abs(y) for e,y in zip(errors,expected) if y!=0),default=0));fm['max_zero_reference_absolute_error']=max(fm['max_zero_reference_absolute_error'],max((e for e,y in zip(errors,expected) if y==0),default=0));fm['max_budget_fraction']=max(fm['max_budget_fraction'],norm);fm['near_zero_reference_count_abs_below_1e_6']+=sum(abs(y)<1e-6 for y in expected)
   inp=inputs[ci];assert actual['source_echo']==inp['source'] and actual['sigma_echo']==inp['sig'] and actual['initial_flux_echo']==inp['flux'] and actual['initial_state_echo']==inp['state'] and actual['ids_echo']==[Q,FAI] and actual['guards']==[16384]*112
   for region in range(R):
    for axial in range(F):
     if (region,axial)!=(Q,FAI):
      first=(region*F+axial)*G;assert actual['fine_flux'][first:first+G]==inp['flux'][first:first+G]
   maxcase=0.;minsource=math.inf;balance=0.;positive=True
   for g in range(G):
    q0,q1,q2,sigma=[actual[z][g] for z in ['q0','q1','q2','sigma']];before=c['before_last_state'][g];expected,gap=quadrature(q0,q1,q2,sigma,before);maxgap=max(maxgap,gap)
    for x,y in zip((actual['state'][g],actual['flux_integral'][g]),expected):
     err=abs(x-y);assert err<=2e-5+2e-5*abs(y);maxcase=max(maxcase,err)
    Qint=q0*L+q1*mu*L*L/2+q2*mu2*L**3/3
    residual=abs(sigma*actual['flux_integral'][g]-Qint-before+actual['state'][g]);balance=max(balance,residual);assert residual<2e-4
    points=[q0,q0+q1*mu*L+q2*mu2*L*L]
    if q2>0:
     vertex=-q1*mu/(2*q2*mu2)
     if 0<vertex<L:points.append(q0+q1*mu*vertex+q2*mu2*vertex*vertex)
    minsource=min(minsource,min(points));assert min(points)>=-2e-5
    assert actual['state'][g]>=-2e-5 and actual['flux_integral'][g]>=-2e-5
    assert actual['fine_flux'][(Q*F+FAI)*G+g]>=inp['flux'][(Q*F+FAI)*G+g]-2e-5
   if mode==4:assert actual['state'][0]==0 and actual['flux_integral'][0]==0
   if mode==5:
    assert all(abs(x-y)<=2e-5+2e-5*abs(y) for x,y in zip(actual['state'],inp['state']))
   maxquadrature=max(maxquadrature,maxcase);maxbalance=max(maxbalance,balance);cases.append(dict(case=ci+1,quadrature_max_absolute_error=maxcase,moment_balance_max_absolute_residual=balance,actual_fitted_polynomial_minimum=minsource,all_actual_inputs_guards_untargeted_flux_exact=True,nonnegative_transport_and_tallies=True))
  assert gp==len(gold);rows.append(dict(model=model,trial=t['trial'],artifact=dict(path=str(f.relative_to(r)),sha256=sha(f),bytes=f.stat().st_size),cases=cases,all_pass=True))
result=dict(benchmark='simplemoc',configuration_id=d['configuration_id'],native_reference_repaired=True,configuration_role='corrected_reference_secondary',all_pass=True,all20_checked=True,artifact_hash_classes=len(hashes),artifact_values=19187,input_prefix_values=6023,guard_words_per_trial=672,field_max_absolute_error=fieldmax,max_oracle_budget_fraction=maxnormalized,independent_direct_Simpson_max_absolute_error=maxquadrature,Simpson128_256_max_difference=maxgap,moment_balance_max_absolute_residual=maxbalance,oracle_scope='Decimal70momentrecurrence+independentSimpsonofactualsavedfittedcoefficientsandexpectedbeforelaststate; actualGPUbeforelasttrajectorynotcaptured. Balanceusesindependentexpectedprelaststate, notclaimedrawobservedtrajectory.',source_scope='Singleactivesegment,conditionaltwo-expressionmomentrepair; interioraxialfitpolicy/placeholdersunchanged, notfullreactorphysics/correlatedmultisegmentconcurrency.',input_guard_and_untargeted_flux_exact=True,result_sha256=sha(bundle/'result.json'),config_sha256=sha(bundle/'config.json'),rows=rows)
result['field_error_metrics']=fieldmetrics;result['tiny_field_precision_scope']='Absolute-plus-relative engineering acceptance; cancellation-scale q1/t2 components may have large relative error. Full perfield nonzero-relative/zero-reference absolute metrics retained; do not interpret maxabsolute accuracy as uniform relative accuracy.'
(p/'output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['rows','field_error_metrics']})
