"""Independent Decimal transport convolution/flux-integral moments.

The source's axial fitting conventions and placeholder constants are retained.
This oracle verifies attenuation of that declared polynomial, not a full reactor
transport or axial-reconstruction model.
"""
from pathlib import Path
from decimal import Decimal as D,localcontext
import json,struct
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',float(x)))[0]
dec=lambda x:D.from_float(f32(x))
pre=[6.];gold=[];cases=[]
with localcontext() as ctx:
 ctx.prec=70;dz=dec(.1);zin=dec(.3);mu=dec(.9);mu2=dec(.3);L=dec(.7);weight=dec(.5)
 for ci,(G,Q,FAI,repeat,mode) in enumerate([(1,1,0,1,0),(17,0,4,2,1),(128,1,2,3,2),(33,0,0,2,3),(65,1,2,3,4),(16,0,3,2,5)]):
  R=2;F=5;source=[];flux=[f32(.25+(i%9)/32) for i in range(R*F*G)];sigma=[f32(1+(g%5)/4) for r in range(R) for g in range(G)];state=[f32(0 if mode==4 and g==0 else .125+(g%13)/32) for g in range(G)]
  if mode==5:state=[f32(.25+(g%2)/4) for g in range(G)]
  for r in range(R):
   for j in range(F):
    for g in range(G):
     base=.5+r/4+(g%7)/16
     s=base if mode==0 else base+j/32 if mode==1 else base+j/16+j*j/32 if mode==2 else base+j*((g%5)-2)/64 if mode==3 else 0 if mode==4 else sigma[r*G+g]*state[g]
     source.append(f32(s))
  pre.extend([R,F,G,Q,FAI,repeat,mode]+source+flux+sigma+state)
  final_flux=[D.from_float(x) for x in flux];outstate=[D.from_float(x) for x in state];scratch=[D(0)]*(14*G);before_last=[];positivemin=[];balance=[]
  for step in range(repeat):
   before_last=list(outstate)
   for g in range(G):
    y2=D.from_float(source[(Q*F+FAI)*G+g])
    if FAI==0:
     y3=D.from_float(source[(Q*F+1)*G+g]);c0=y2;c1=(y3-y2)/dz;c2=D(0)
    elif FAI==F-1:
     y1=D.from_float(source[(Q*F+FAI-1)*G+g]);c0=y2;c1=(y2-y1)/dz;c2=D(0)
    else:
     y1=D.from_float(source[(Q*F+FAI-1)*G+g]);y3=D.from_float(source[(Q*F+FAI+1)*G+g]);c0=y2;c1=(y1-y3)/(2*dz);c2=(y1-2*y2+y3)/(2*dz*dz)
    q0=c0+c1*zin+c2*zin*zin;q1=c1+2*c2*zin;q2=c2;sig=D.from_float(sigma[Q*G+g]);tau=sig*L;e=(-tau).exp();ev=1-e;st=outstate[g]
    # Independent convolution basis moments via integration-by-parts recurrence:
    # H_j = integral_0^L exp(-sigma*(L-s))*s^j ds.
    H0=ev/sig;H1=(L-H0)/sig;H2=(L*L-2*H1)/sig
    sout=st*e+q0*H0+q1*mu*H1+q2*mu2*H2
    integral=st*H0+q0*(L-H0)/sig+q1*mu*(L*L/2-H1)/sig+q2*mu2*(L**3/3-H2)/sig
    reused=H2;sig2=sig*sig;tally=weight*integral;t1=q0*H0;t2=q1*mu*H1;t3=q2*mu2*H2;t4=st*e
    vals=[q0,q1,q2,sig,tau,sig2,ev,reused,integral,tally,t1,t2,t3,t4]
    for j,v in enumerate(vals):scratch[j*G+g]=v
    outstate[g]=sout;final_flux[(Q*F+FAI)*G+g]+=tally
    candidates=[q0,q0+q1*mu*L+q2*mu2*L*L]
    if q2>0:
     vertex=-q1*mu/(2*q2*mu2)
     if 0<vertex<L:candidates.append(q0+q1*mu*vertex+q2*mu2*vertex*vertex)
    positivemin.append(float(min(candidates)));assert min(candidates)>=0
    Qint=q0*L+q1*mu*L*L/2+q2*mu2*L**3/3;balance.append(float(sig*integral-Qint-st+sout))
  vals=list(map(float,final_flux+outstate+scratch))+source+sigma+flux+state+[Q,FAI]+[16384]*112;gold.extend(vals)
  cases.append(dict(case=ci+1,regions=R,fine_intervals=F,energy_groups=G,QSR=Q,FAI=FAI,repeat=repeat,mode=mode,source_values=len(source),cross_sections=len(sigma),oracle_values=len(vals),source_polynomial_minimum=min(positivemin),independent_balance_max=max(map(abs,balance)),before_last_state=list(map(float,before_last)),final_state=list(map(float,outstate)),physical_control='constant_positive' if mode==0 else 'affine_last_boundary' if mode==1 else 'quadratic_interior' if mode==2 else 'signed_affine_first_boundary_positive_source' if mode==3 else 'zero_source_pure_absorption' if mode==4 else 'constant_source_equilibrium'))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]: (p/name).write_bytes(struct.pack('<'+str(len(a))+'d',*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_values=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),independent_oracle='Decimal70 linearODE convolution andfluxintegral moments byintegration-by-parts recurrence, notexpandednativeformula; exactsourceF32inputs/constants withretainedaxialfitpolicy. Positiveemissionpolynomial/absorption/equilibrium controls andmomentbalance.',correction='Quadraticstatereuseproperparentheses andhalflinearfluxintegral factor; originalCPU/GPUcounterexamples preserved. Nativeconditionalcorrectedonly.',budget=dict(atol=2e-5,rtol=2e-5,scope='EngineeringFloat32operatorbudgetoverboundedsigma1..2/tau.7..1.4andmoderatepolynomialcoefficients/repeat<=3, notformalroundoffbound. Fullperfielderrorsnormalizedtolretained; tinyfieldsabsolute-accuracyonly.'),support='OneactivesegmentassourceVERIFY; original128blocksx128threads overlaunch/serializedenergyloop/global14scratch retained. >1segmentglobalsharedstate/scratchrace uncertified, notrunbadGPU; sourcefittedinteriorc1signretained, noaxialreconstruction/whole-reactormodelclaim.'),indent=2)+'\n')
print(len(pre),len(gold),len(pre)+len(gold),min(c['source_polynomial_minimum'] for c in cases))
