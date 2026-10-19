from decimal import Decimal as D,localcontext
from pathlib import Path
import json,hashlib,re
p=Path(__file__).resolve().parent;r=p.parent.parent;src=r/'src/rushlarsen-cuda'
with localcontext()as ctx:
 ctx.prec=80
 V=D('-85.423');F=D('96485.3415');R=D('8314.472');T=D(310);g=D('.35');Cao=D(2);Nai=D('10.132');Nao=D(140);Cai=D('.000153');K=D(1000);sat=D('.1');KmCa=D('1.38');KmNai=D('87.5');alpha=D('2.5');c=(Cao+KmCa)*(KmNai**3+Nao**3)
 def terms(v):
  a=Cao*Nai**3*(F*g*v/(R*T)).exp()-alpha*Nao**3*Cai*(F*(g-1)*v/(R*T)).exp();s=1+sat*(F*(g-1)*v/(R*T)).exp();ap=Cao*Nai**3*F*g/(R*T)*(F*g*v/(R*T)).exp()-alpha*Nao**3*Cai*F*(g-1)/(R*T)*(F*(g-1)*v/(R*T)).exp();sp=sat*F*(g-1)/(R*T)*(F*(g-1)*v/(R*T)).exp();return a,s,ap,sp
 a,s,ap,sp=terms(V);correct=K*(ap*s-a*sp)/(s*s*c);native=K*ap/(s*c)-K*a*sp*c/(s*s)
 fun=lambda v:(lambda z:K*z[0]/(z[1]*c))(terms(v));diffs=[{'h':str(h),'centered_derivative':str((fun(V+h)-fun(V-h))/(2*h)),'abs_error':str(abs((fun(V+h)-fun(V-h))/(2*h)-correct))}for h in [D('1e-3'),D('1e-4'),D('1e-5')]]
 info={'benchmark':'rushlarsen','witness':'Independent80decimaldigitquotientrule for native i_NaCa(V) with allotherstates/parametersfrozen','equation':'I(V)=K*A(V)/(S(V)*C), Cconstant; Iprime=K*(Aprime*S-A*Sprime)/(S*S*C)','source_problem':'Native secondquotientterm multipliesC ratherthandividingC; correctedownedcopy changesonlytwo*to/ inthatsecondterm','default_V':str(V),'constant_C':str(c),'native_expression_derivative':str(native),'independent_derivative':str(correct),'centered_difference_controls':diffs,'native_source_immutable':True,'original_cpu_gpu_one_step_logs':[],'scope':'LocalJacobian/finiteone-stepwitness only, notcompletecardiacmodelparameter/longtrajectoryvalidation'}
 for name in ['original_cpu_one_step.log','original_gpu_one_step.log']:
  f=r/'.porting-state/portable_julia/rushlarsen_readiness'/name
  if f.exists():info['original_cpu_gpu_one_step_logs'].append({'path':str(f.relative_to(r)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'text':f.read_text()})
 (p/'reference_derivative_witness.json').write_text(json.dumps(info,indent=2)+'\n')
for name,new in [('kernels.cu','study_corrected_kernel.h'),('reference.cu','study_corrected_reference.h')]:
 original=(src/name).read_text();matches=list(re.finditer(r'const double di_NaCa_dV =.*?;',original,re.S));assert len(matches)==1;mt=matches[0];chunk=mt.group();needle=')*(Ca_o + Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) +';assert chunk.count(needle)==1;fixed=chunk.replace(needle,')/(Ca_o + Km_Ca)/((Km_Nai*Km_Nai*Km_Nai) +');(p/new).write_text(original[:mt.start()]+fixed+original[mt.end():])
print('WITNESS',info['native_expression_derivative'],info['independent_derivative']);print('minimalcorrectiontwooperatorsonly')
