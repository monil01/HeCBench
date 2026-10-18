"""Independent SRAD oracle: edge-flux updates and mathematical sum workspaces."""
from pathlib import Path
import math,struct,json
p=Path(__file__).resolve().parent;pre=[4.];gold=[];cases=[]
def workspace(v):
 a=list(v);no=len(a);mul=1
 while True:
  blocks=(no+255)//256;before=list(a)
  for block in range(blocks):a[block*mul*256]=math.fsum(before[i*mul] for i in range(block*256,min((block+1)*256,no)))
  if blocks==1:return a
  no=blocks;mul*=256
for ci,(nr,nc,steps,lam) in enumerate([(16,16,2,.5),(19,27,3,.5),(3,257,2,.125),(257,257,3,.75)]):
 N=nr*nc;raw=[float(15+((row*37+col*13+((row//4+col//3)%3)*71+row*col*3+ci*11)%226)) for col in range(nc) for row in range(nr)]
 iN=[max(i-1,0) for i in range(nr)];iS=[min(i+1,nr-1) for i in range(nr)];jE=[min(j+1,nc-1) for j in range(nc)];jW=[max(j-1,0) for j in range(nc)];neighbors=iN+iS+jE+jW
 pre.extend([nr,nc,steps,lam]+raw+neighbors)
 J=[math.exp(x/255.) for x in raw];extracted=list(J);initialsum=math.fsum(J);initialmin=min(J);initialmax=max(J);stats=[];levels=0;no=N
 while True:
  levels+=1;no=(no+255)//256
  if no==1:break
 for it in range(steps):
  lastpre=list(J);total=math.fsum(J);total2=math.fsum(x*x for x in J);mean=total/N;var=total2/N-mean*mean;q0=var/(mean*mean);assert var>1e-3 and q0>1e-3;stats.extend([total,total2,mean,var,q0]);dN=[];dS=[];dE=[];dW=[];coeff=[]
  for col in range(nc):
   for row in range(nr):
    at=col*nr+row;center=J[at];north=J[col*nr+iN[row]]-center;south=J[col*nr+iS[row]]-center;east=J[jE[col]*nr+row]-center;west=J[jW[col]*nr+row]-center
    g2=math.fsum(x*x for x in [north,south,east,west])/(center*center);lap=math.fsum([north,south,east,west])/center;qsqr=(.5*g2-.0625*lap*lap)/((1+.25*lap)**2);den=(qsqr-q0)/(q0*(1+q0));c=min(max(1/(1+den),0.),1.);dN.append(north);dS.append(south);dE.append(east);dW.append(west);coeff.append(c)
  # Independent conservative pairwise edge assembly, not the native pointwise
  # divergence expression. Every interior edge contributes equal/opposite flux.
  divergence=[0.]*N
  for col in range(nc):
   for row in range(nr):
    at=col*nr+row
    if row+1<nr:
     other=at+1;f=coeff[other]*(J[other]-J[at]);divergence[at]+=f;divergence[other]-=f
    if col+1<nc:
     other=at+nr;f=coeff[other]*(J[other]-J[at]);divergence[at]+=f;divergence[other]-=f
  J=[x+.25*lam*div for x,div in zip(J,divergence)]
  assert min(J)>=initialmin-1e-12 and max(J)<=initialmax+1e-12 and abs(math.fsum(J)-initialsum)<1e-8
 sums=workspace(lastpre);sums2=workspace([x*x for x in lastpre]);compressed=[255.*math.log(x) for x in J]
 vals=compressed+J+sums+sums2+dN+dS+dE+dW+coeff+raw+extracted+neighbors+stats+[16384]*192;gold.extend(vals)
 cases.append(dict(case=ci+1,nr=nr,nc=nc,pixels=N,steps=steps,lambda_=lam,reduction_levels=levels,compute_launches=steps*(3+levels),all_kernel_roles=6,oracle_values=len(vals),raw_input_min=min(raw),raw_input_max=max(raw),extracted_input_min=initialmin,extracted_input_max=initialmax,initial_extracted_mass=initialsum,final_extracted_mass=math.fsum(J),minimum_q0sqr=min(stats[4::5]),minimum_variance=min(stats[3::5]),statistic_values=len(stats),positive_nonzero_variance_scope=True))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]: (p/name).write_bytes(struct.pack('<'+str(len(a))+'d',*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_values=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),independent_oracle='Float64globalmath.fsumstatistics, fullcolumnmajorreflectinggradient/coefficientfields, conservativepairwiseedgefluxassemblyratherthanpointwisestencil, mathematicallevelnodes/workspace sums, exp/logimageconversion. Allpixels/internalarrays checked.',budget=dict(atol=1e-4,rtol=1e-5,scope='PredeclaredFloat32engineeringoperatorbudgetacross<=3steps/nonzerovariance; notformalforward-errorbound. Pointwisecompressederrorswellbelow1graylevelifaccepted, rawinternaltotalsscaledrelative. Inputs/neighbor/guards exact.'),support='Fourdeterministicnonuniformimages, λ<=.75, variance/q0>1e-3throughout. Uniform/nearzero-varianceimage leadsunhandledq0divisionzero inoriginaloperator, notcertified/notframeworkunsupported. NoPGMresize/inputfileIO/1000stepclinicalimagequalityclaim.',native_reference='Conditionaltwoindexreadcorrection inpartialreductionremainder, allother5kernelsunchanged. OriginalGPUfullstagecounterexample retained.'),indent=2)+'\n')
print(len(pre),len(gold),len(pre)+len(gold),sum(c['compute_launches'] for c in cases))
