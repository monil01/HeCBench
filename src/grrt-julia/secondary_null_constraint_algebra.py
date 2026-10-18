"""CPU-only exact rational witness; prints evidence without changing saved captures.
A=0, K=ptheta**2+L**2/sin(theta)**2 constant, away from artificial polar clamp.
This assesses the null-sector off-constraint extension, not the physical model.
"""
from fractions import Fraction as F
import json
cases=[]
for r in map(F,('5/2','3','6','20','1000')):
 for p in map(F,('1/2','1','3/2')):
  for k in map(F,('0','1','5')):
   f=(r-2)/r;fp=2/r**2;h=-1/f+f*p*p+k/r**2
   dr=-f*p;dhdr=fp/f**2+fp*p*p-2*k/r**3;dhdp=2*f*p
   source=((r-1)*k-2*r**3)/(r**3*(r-2))+2*p*p*(r-1)/r**2
   canonical=1/(r*r*f*f)+p*p/(r*r)-k/r**3
   hs=dhdr*dr+dhdp*source;hc=dhdr*dr+dhdp*canonical
   assert hs==2*p*(2*r-3)*h/r**2
   assert hc==0
   assert canonical-source==-(2*r-3)*h/(r*(r-2))
   cases.append({'r':str(r),'pr':str(p),'K':str(k),'H':str(h),'source_dH':str(hs),'canonical_dH':str(hc)})
print(json.dumps({'cases_checked_exact_rational':len(cases),'equivalence_condition':'H=0, A=0, K=kappa, no polar clamp activation','cases':cases},indent=2))
