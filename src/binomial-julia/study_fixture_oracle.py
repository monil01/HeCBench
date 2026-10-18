from pathlib import Path
import math, struct, json
p=Path(__file__).resolve().parent
records=[];pre=[];oracle=[];bs=[]
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
for n in [16,129]:
 for i in range(n):
  S=f(5+(i*17%101)/4);X=f(1+(i*29%197)/2);T=f(.25+(i*7%39)/4);rate=f(.06);vol=f(.1 if i%3==0 else .2 if i%3==1 else .35)
  # source-style FP32 coefficient preprocessing explicitlyserialized
  dt=f(T/f(2048));vdt=f(vol*f(math.sqrt(dt)));rdt=f(rate*dt);u=f(math.exp(vdt));down=f(math.exp(-vdt));If=f(math.exp(rdt));Df=f(math.exp(-rdt));pu=f(f(If-down)/f(u-down));pd=f(1-pu);a=f(pu*Df);b=f(pd*Df)
  records.extend([S,X,T,rate,vol]);pre.extend([S,X,vdt,a,b])
  terms=[]
  for k in range(2049):
   payoff=max(S*math.exp(vdt*(2*k-2048))-X,0)
   if payoff:terms.append(math.exp(math.lgamma(2049)-math.lgamma(k+1)-math.lgamma(2049-k)+k*math.log(a)+(2048-k)*math.log(b))*payoff)
  oracle.append(math.fsum(terms));d1=(math.log(S/X)+(rate+.5*vol*vol)*T)/(vol*math.sqrt(T));d2=d1-vol*math.sqrt(T);N=lambda x:.5*(1+math.erf(x/math.sqrt(2)));bs.append(S*N(d1)-X*math.exp(-rate*T)*N(d2))
for name,data,fmt in [('fixtures_f32.bin',records+pre,'f'),('oracle_f64.bin',oracle,'d')]:
 with(p/name).open('wb')as io:io.write(struct.pack('<'+fmt*len(data),*data))
(p/'mathematical_oracle.json').write_text(json.dumps({'method':'DirectterminalbinomialexpectationviaFloat64lgamma/logweightsandfsum; no backwardrecursion/sharedexchange reused','price_count':len(oracle),'coefficient_contract':'ExactserializedpreprocessedFloat32coefficients, discounta+b retained ratherthan reinterpreted as exactanalyticrate','max_abs_discrete_to_continuous_BS':max(abs(a-b)for a,b in zip(oracle,bs)),'continuous_BS_is_not_acceptance':'Finite2048stepdiscretizationandroundedcoefficienteffects remain explicit','bounds':'Allpayoffs/weightsnonnegative; alloracleprices>=0'},indent=2)+'\n')
