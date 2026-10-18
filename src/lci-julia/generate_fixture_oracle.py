from pathlib import Path
from decimal import Decimal,localcontext
import math,struct,json,hashlib,random
B=Path(__file__).resolve().parent
D=Decimal.from_float
with localcontext() as ctx:
 ctx.prec=80
 alpha=[Decimal(math.factorial(2*l))/Decimal(2**l*math.factorial(l)**2) for l in range(193)]
 af=[float(x) for x in alpha]
 rng=random.Random(64)
 cases=[dict(name='source_style_dense',t=.2,c=[.5]+[rng.uniform(-6,6) for _ in range(63)]+[0.]),dict(name='equilibrium_zero_moments',t=1.,c=[.75]+[0.]*64),dict(name='moment1_dyadic',t=200.,c=[.25,1.]+[0.]*63),dict(name='moment63_signed_dyadic',t=17.5,c=[.625]+[0.]*62+[-.5,0.])]
 pi=D(math.pi);kappa=pi*pi/3-2*D(1.2020569031595942);theta=5/(2*pi**4)
 outputs=[]
 for case in cases:
  c=list(map(D,case['c']));t=D(case['t']);ans=[-c[0]/3/t*(1+D(.1)*c[1])]
  for l in range(1,64):
   ld=Decimal(l);bb=2*(14*ld*ld+7*ld-2)/(4*ld-1)/(4*ld+3)-Decimal(4)/3
   u=-(2*ld-1)*(2*ld+1)*(2*ld+2)/(4*ld+3)/(4*ld+5)
   cv=(2*ld-1)*2*ld*(2*ld+2)/(4*ld-3)/(4*ld-1)
   lhs=(u*c[l+1]+bb-Decimal(2)/15*c[1]+cv*(c[l-1] if l>1 else 1))/t
   s1=Decimal(0)
   for m in range(1,64):
    if not c[m]:continue
    for n in range(1,64):
     if not c[n] or abs(m-n)>=l+1 or m+n-l<0:continue
     omega=alpha[m-n+l]*alpha[m+n-l]*alpha[n-m+l]/alpha[m+n+l]*Decimal(4*l+1)/Decimal(2*(n+m+l)+1)
     s1+=omega*c[m]*c[n]
   s2=sum((c[n]*c[n]/Decimal(4*n+1) for n in range(1,64)),Decimal(0))*Decimal((2*l-1)*(l+1))*c[l]/3
   ans.append(-lhs-c[0]*theta*((kappa+pi*pi*l*(2*l+1)/3)*c[l]+kappa*s1+kappa*s2))
  outputs.extend(map(float,ans))
 for name,vals in [('inputs_f64.bin',sum((a['c'] for a in cases),[])),('alpha_f64.bin',af),('oracle_f64.bin',outputs)]:
  (B/name).write_bytes(struct.pack('<'+'d'*len(vals),*vals))
 files={n:dict(bytes=(B/n).stat().st_size,sha256=hashlib.sha256((B/n).read_bytes()).hexdigest()) for n in ['inputs_f64.bin','alpha_f64.bin','oracle_f64.bin']}
 (B/'fixture_provenance.json').write_text(json.dumps(dict(cases=[{k:v for k,v in a.items() if k!='c'} for a in cases],files=files,alpha_method='Exact integer factorial ratio (2l)!/(2^l*(l!)^2) rounded via Decimal80 toFloat64; native correctedconstant/Julia actualinputsamebytes',oracle='IndependentDecimal80 momentalgebra/exactinteger-factorial ratios, sourceactualFloat64pi/zeta constants; no continuum/ODE validation',guard=12345.5,atol=1e-8,rtol=1e-10,original_table_domain='fact171/doublefact298; alpha(l>=150) outofbounds; originalsum maxm+n+l189',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
