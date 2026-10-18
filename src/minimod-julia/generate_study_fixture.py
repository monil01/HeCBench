"""Independent global finite-difference acoustic/PML oracle; no shared-tile reuse."""
from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent;R=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];N=24;LDX,LDY,LDZ=40,32,32;SIZE=LDX*LDY*LDZ;LEAD=20;TAIL=8
ind=lambda i,j,k:((i+4)*LDY+j+4)*LDZ+k+4
coef=[R(R(a/b)/400) for a,b in [(-205,72),(8,5),(-1,5),(8,315),(-1,560)]];c0=R(R(coef[0]+coef[0])+coef[0]);hd=R(1/1600);pre=[3,N,LDX,LDY,LDZ,LEAD,TAIL,c0]+[coef[r] for r in range(1,5) for _ in range(3)]+[hd,hd,hd];gold=[];cases=[]
active=[ind(i,j,k) for i in range(N) for j in range(N) for k in range(N)];active_set=set(active);pml=lambda i,j,k:not(8<=i<16 and 8<=j<16 and 8<=k<16)
for ci,steps in enumerate([4,2,1]):
 u=[0.]*SIZE;v=[0.]*SIZE;phi=[0.]*SIZE;vp=[0.]*SIZE;eta=[0.]*SIZE
 for i in range(-4,36):
  for j in range(-4,28):
   for k in range(-4,28):
    z=ind(i,j,k);poly=(i*i+2*j*j+3*k*k)/4096+i/128-j/256
    u[z]=v[z]=R(poly) if ci==1 else 0.;vp[z]=R(4*(1+(i%5)/32)) if ci==0 else R(4)
    damp=sum((max(8-t,t-15,0)/8)**2 for t in [i,j,k]);eta[z]=R(min(damp,9)/64) if ci!=1 else 0.
 source=[.5,.125,.0625,-.03125,0.][:steps+1] if ci==0 else [.5]+[0.]*steps
 initial=[a.copy() for a in [u,v,phi,vp,eta]];pre +=[steps]+source
 for a in initial:pre+=a
 first_lap_err=0.;stage_max=[]
 for step in range(1,steps+1):
  # u andprevious-levelv areindependent; disjointPML/innerregionscoverallactivecells.
  for i in range(N):
   for j in range(N):
    for k in range(N):
     z=ind(i,j,k);lap=c0*u[z]
     for r in range(1,5):
      for axis in range(3):
       d=[0,0,0];d[axis]=r;lap+=coef[r]*(u[ind(i+d[0],j+d[1],k+d[2])]+u[ind(i-d[0],j-d[1],k-d[2])])
     if ci==1 and step==1:first_lap_err=max(first_lap_err,abs(lap-12/1638400))
     if pml(i,j,k):
      e=eta[z];v[z]=((2*e+2-e*e)*u[z]+(vp[z]*(lap+phi[z])-v[z]))/(2*e+1)
      grad=0.
      for axis in range(3):
       d=[0,0,0];d[axis]=1;zp=ind(i+d[0],j+d[1],k+d[2]);zm=ind(i-d[0],j-d[1],k-d[2]);grad+=(eta[zp]-eta[zm])*(u[zp]-u[zm])*hd
      phi[z]=(phi[z]-grad)/(1+e)
     else:v[z]=2*u[z]+vp[z]*lap-v[z]
  v[ind(12,12,12)]+=source[step];u,v=v,u;stage_max.append(max(abs(u[z]) for z in active));assert all(math.isfinite(x) for x in u+v+phi)
 if ci==1:assert first_lap_err<2e-8
 fields=[u,v,phi,vp,eta]
 for group in [fields,initial]:
  for a in group:gold +=[-12345.]*LEAD+a+[-12345.]*TAIL
 cases.append(dict(case=ci,steps=steps,source=source,initial_condition=['point_forcing_with_variable_vp_and_PML','manufactured_quadratic_zero_PML','zero_solution_with_nonzero_PML'][ci],first_step_quadratic_laplacian_abs_error=first_lap_err if ci==1 else None,expected_quadratic_analytic_laplacian=12/1638400 if ci==1 else None,stage_max_abs_u=stage_max,final_nonzero_phi=sum(x!=0 for x in phi),atol=2e-6,rtol=2e-5,zero_solution_exact=ci==2))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
meta=dict(field_budgets={'u_v':{'atol':2e-6,'rtol':2e-5},'phi':{'atol':1e-10,'rtol':2e-5}},N=N,payload=SIZE,leading_guards=LEAD,trailing_guards=TAIL,source_leadpad_policy='20floats =32-((lz4+ndamp8)&31), source(0,0,ndampz)128B-alignment retained; added8trailingguards',cases=cases,input_prefix_count=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),atol=2e-6,rtol=2e-5,oracle='Independent directglobal eighth-order25point acoustic/PML finite-difference recurrence Float64 withactualF32inputs/coefficients; fulltwolevels/phi/immutablevpeta andinputecho. QuadraticanalyticLaplacian andzero controls supplementdiscreteoracle.',source_scope='N24/PML8validalignedslabs,7regions+onepointsource/update andlevels swap. Sourcearraypaddedsteps+1withsource0unused, actualsource[1..steps] conventionexplicit, conditionalhostinitializationrepair notoriginalCLI. Fullyspecifiedinitialwave/phi/vp/eta/halos.',exact_masks='Allleading/trailingguards, immutablevpeta, allpreexecutioninputbytes, everyu/voutsidephysicalvolume, phioutsidePML unchangedexact; zero-caseentirefields exact. Floatingbudget onlymodifiedwave/PMLfields. RemainingvalueencodingFloat64doesnotmeanFloat64GPU.')
(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k!='cases'}));print(cases)
