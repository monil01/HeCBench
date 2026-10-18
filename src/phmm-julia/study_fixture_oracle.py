"""Exact-rational oracle for the source's defined-storage stale-buffer recurrence."""
from pathlib import Path
from fractions import Fraction as Q
import math,struct,json,hashlib
P=Path(__file__).resolve().parent
NX,NY,B,S=11,40,4,2;NF=12*41*4*2;NT=12*4*2*3
idx=lambda i,j,b,s:((i*41+j)*4+b)*2+s
tr=lambda i,b,k,s:((i*4+b)*2+k)*3+s
li=lambda i,j,b,s:((i*2+j)*4+b)*2+s
f=[3.];out=[];budgets=[];case_meta=[];smallest=Q(1);largest=Q(0)
for repeat,zero in [(1,False),(2,False),(1,True)]:
 a=[Q(1+(i*7+i//11)%15,16)for i in range(NF)]
 e=[Q(0)if zero else Q(1+(i*5+i//13)%7,8)for i in range(NF)]
 t=[Q(1+(i*11+i//7)%3,16)for i in range(NT)]
 l=[Q(1+(i*7+i//3)%3,8)for i in range(32)];st=[Q(1+i%3,8)for i in range(8)]
 f +=[float(repeat),float(zero)]+[float(x)for v in [a,e,t,l,st]for x in v]
 buffers=[a.copy(),a.copy()];written=[set(),set()];source=0
 for rep in range(repeat):
  for i in range(1,12):
   for j in range(1,41):
    target=1-source
    for b in range(4):
     for s in range(2):
      vals=[]
      for ip,jp,ti,lj in [(i-1,j-1,i-1,(0,0)),(i-1,j,i-1,(0,1)),(i,j-1,i,(1,0)),(i,j,i,(1,1))]:
       products=[buffers[source][idx(ip,jp,b,k)]*t[tr(ti,b,k,s)]for k in range(2)]
       terms=sum(products)*l[li(*lj,b,s)];vals+= [terms]
       for v in products+[terms]:
        if v:smallest=min(smallest,v)
        largest=max(largest,v)
      val=sum(vals)*e[idx(i,j,b,s)];buffers[target][idx(i,j,b,s)]=val;written[target].add(idx(i,j,b,s))
      if val:smallest=min(smallest,val)
      largest=max(largest,val)
    source=target
 u=Q(1,2**53);n=32*440*repeat+16;gamma=Q(n)*u/(1-Q(n)*u);localout=[];localbudget=[]
 for bi,v in enumerate(buffers):
  localout +=[-12345.]+[float(x)for x in v]+[-12345.]
  localbudget +=[0.]+[math.nextafter(float((gamma+2*u)*abs(x)),math.inf)if k in written[bi]and x else 0. for k,x in enumerate(v)]+[0.]
 for v in [e,t,l,st]:localout +=[-12345.]+[float(x)for x in v]+[-12345.];localbudget +=[0.]*(len(v)+2)
 out+=localout;budgets+=localbudget
 case_meta.append({'repeat':repeat,'zero_emission':zero,'timed_kernel_calls':440*repeat,'A_written_cells':len(written[0])//8,'B_written_cells':len(written[1])//8,'positive_initial_bound':float(min(a)),'minimum_nonzero_final':min(float(x)for v in buffers for x in v if x),'maximum_final':max(float(x)for v in buffers for x in v),'relative_forward_bound_gamma_plus_2u':float(gamma+2*u)})
assert smallest>Q(1,2**1022)and largest<2**1023
pack=lambda v:struct.pack('<%dd'%len(v),*v)
for name,data in [('fixtures_f64.bin',f),('oracle_f64.bin',out),('error_budget_f64.bin',budgets)]: (P/name).write_bytes(pack(data))
(P/'independent_oracle.json').write_text(json.dumps({'cases':case_meta,'prefix_words':len(f),'output_words':len(f)+len(out),'oracle_suffix_words':len(out),'method':'ExactPythonFraction sourcecoefficient4-neighbor+selfterm recurrence, independenthigherprecisionserialstate-mixing replay, bothphysicalmatrices fullydefinedB=Ainitial. Allreadonlyarrays/unusedthirdtransitionstates/canariesexact.','minimum_nonzero_intermediate':float(smallest),'maximum_intermediate':float(largest),'source_case_scope':'Fullsource11x40cells/batch4/twostates, repeats1/2 andzeroemissioncontrol1; original1000repeatCLI notverified. Dormantj0notcalled.','scientific_limit':'NotstandardPairHMMprobabilitymodelcertificate; sourcepartial-update/stale-buffer/selfterm/nonnormalizedcoefficients explicitlypreserved.','reference_correction':'HostsecondfullmatrixinitializedfromA beforeROI, originalundefinedCLI andsourcewrittenmask/poisonwitness retained.'},indent=2)+'\n')
(P/'error_model_derivation.json').write_text(json.dumps({'declared_before_measurement':True,'arithmetic':'Binary64positiveops, exactdyadicinputs, roundtonearesteven inclFMA contraction (fewerroundings), no transcendental/division inGPUkernel','bound':'Sourcevalueupdate has<=24multiply/addroundings. Positiveweightedlinearcombination propagatesrelativeerror withoutcancellation; entire440*repeatlaunchdependencychain bounded conservativelyby gamma_(32*440*repeat+16)+2u, u2^-53. Applycomponentwiseabsolute budget=relativebound*exactFractionvalue; oracleFloat64encoding roundingcovered2u. Untouchedcells/readonly/guards/exactzerosbudget0.','normal_range_witness':{'minimum_positive_intermediate':float(smallest),'maximum_intermediate':float(largest)},'limitations':'Finiteexplicitdyadicfixture andsourceoperationgraph only; notnormalizedprobability/continuum/statisticalquality/generalHMMproof. Ownbothrawrefsagreement rtol1e-11 atzeroatol, independentcomponentbudgetsstricter.','source_hashes':{str(q):hashlib.sha256(q.read_bytes()).hexdigest()for q in [P/'study_fixture_oracle.py',P.parent/'phmm-cuda/kernel.h']}},indent=2)+'\n')
print({'prefix_words':len(f),'output_words':len(f)+len(out),'oracle_suffix_words':len(out),'case_info':case_meta})
