"""Independent arbitrary-precision source-formula witness, not reactor-physics certification."""
from pathlib import Path
import struct,json,hashlib
import mpmath as mp
mp.mp.dps=70
p=Path(__file__).resolve().parent;raw=(p/'fixtures_f64.bin').read_bytes();f=struct.unpack('<'+'d'*(len(raw)//8),raw);nc,maxp,maxw,maxn,ncases=map(int,f[:5]);cases=[tuple(map(int,f[5+2*i:7+2*i]))for i in range(ncases)];pos=5+2*ncases
def take(n):
 global pos
 a=f[pos:pos+n];pos+=n;return a
num=list(map(int,take(12)));mats=list(map(int,take(12*maxn)));nw=list(map(int,take(nc)));concs=take(12*maxn);pseudo=take(nc*4);pd=take(nc*maxp*9);wd=take(nc*maxw*5);assert pos==len(f)
poles=[pd[i:i+9]for i in range(0,len(pd),9)];windows=[wd[i:i+5]for i in range(0,len(wd),5)]
A=2806196910506780709;MASK=(1<<63)-1;dist=[.140,.052,.275,.134,.154,.064,.066,.055,.008,.015,.025,.013];branches={'Abrarov':0,'asymptotic':0};max_exp_relative=mp.mpf(0)
def M(x):return mp.mpf(x)
def C(a,b):return mp.mpc(M(a),M(b))
def W(z):
 global max_exp_relative
 if abs(z)<6:
  branches['Abrarov']+=1;arg=12j*z;x=1+arg.real/M(4096)
  for _ in range(12):x=x*x
  expz=x*mp.mpc(mp.cos(arg.imag),mp.sin(arg.imag));max_exp_relative=max(max_exp_relative,abs(x/mp.exp(arg.real)-1))
  value=1j*(1-expz)/(12*z);s=0j
  an=[.2758402,.2245740,.1594149,.09866577,.05324414,.02505215,.01027747,.003676164,.001146494,.0003117570];den=[9.869604,39.47842,88.82644,157.9137,246.7401,355.3058,483.6106,631.6547,799.4380,986.9604]
  for n in range(10):s+=M(an[n])*(((-1 if n%2==0 else 1)*expz)-1)/(M(den[n])-144*z*z)
  return value+mp.mpc(0,M(81.24330))*z*s
 branches['asymptotic']+=1;z2=z*z
 return z*1j*(M(.512424224754768462984202823134979415014943561548661637413182)/(z2-M(.275255128608410950901357962647054304017026259671664935783653))+M(.051765358792987823963876628425793170829107067780337219430904)/(z2-M(2.724744871391589049098642037352945695982973740328335064216346)))
cache={};seed=1070;energies=[];materials=[];seeds=[]
for i in range(max(n for n,d in cases)):
 seed=(A*seed+1)&MASK;energies.append(seed/(1<<63));seed=(A*seed+1)&MASK;roll=seed/(1<<63);seeds.append(seed);mat=0
 for j in range(12):
  running=0.
  for k in range(j,0,-1):running+=dist[k]
  if roll<running:mat=j;break
 materials.append(mat)
min_gap=mp.inf;out=[]
for n,doppler in cases:
 for i in range(n):
  E=M(energies[i]);mat=materials[i]
  key=(i,doppler)
  if key not in cache:
   result=[mp.mpf(0)]*4
   for j in range(num[mat]):
    nuc=mats[mat*maxn+j];window=min(int(E/(1/M(nw[nuc]))),nw[nuc]-1);w=windows[nuc*maxw+window]
    sig=[E*M(w[k])for k in range(3)];sigs=[]
    for l in range(4):
     phi=M(pseudo[nuc*4+l])*mp.sqrt(E)
     if l==1:phi+=mp.atan(phi)
     elif l==2:phi-=mp.atan(3*phi/(3-phi*phi))
     elif l==3:phi-=mp.atan(phi*(15-phi*phi)/(15-6*phi*phi))
     sigs.append(mp.mpc(mp.cos(2*phi),-mp.sin(2*phi)))
    for k in range(int(w[3]),int(w[4])):
     row=poles[nuc*maxp+k];ea=C(*row[:2]);rt=C(*row[2:4]);ra=C(*row[4:6]);rf=C(*row[6:8]);l=int(row[8]);term=W((E-ea)/2)if doppler else (1j/(ea-mp.sqrt(E)))/E
     sig[0]+=(rt*term*sigs[l]).real;sig[1]+=(ra*term).real;sig[2]+=(rf*term).real
    micro=sig+[sig[0]-sig[1]]
    for k in range(4):result[k]+=micro[k]*M(concs[mat*maxn+j])
   ix=max(range(4),key=lambda k:result[k]);s=sorted(result,reverse=True);min_gap=min(min_gap,s[0]-s[1]);cache[key]=(ix+1,[float(x)for x in result])
  ix,xs=cache[key];seed=seeds[i];out +=[ix,energies[i],mat,seed&0xffffffff,seed>>32]+xs
assert all(mp.isfinite(v)for v in out)
(p/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(out),*out));info={'method':'IndependentPythonbigintsequentialLCGstream,70decimaldigitmpmathscalar/complexevaluationof source0K/Doppler formulas inclsourcefast_exp12squarings androundedconstantcoefficients; sameexclusivewindowendpolicy. Not exactphysicalFaddeeva/transportcertification.','oracle_count':len(out),'output_count':len(f)+len(out)+len(f)+70,'input_prefix_count':len(f),'minimum_argmax_gap':str(min_gap),'diagnostic_branch_evaluations':branches,'source_fast_exp_max_relative_difference_vs_exact_exp_in_fixture':str(max_exp_relative),'macroscopic_acceptance':{'atol':1e-9,'rtol':1e-8,'predeclared':True,'rationale':'FP64finiteboundednamedsourceformulaevaluation withSFUcomplexrationalarithmetic; 70digitoracle and generous1e-8relative+1e-9absolute namedoperatorbudget, not certifiedgeneralforwarderrorbound orphysicalmodelaccuracy.'},'exact_fields':'Alltimedargmaxindices, energy/material/seed32bithalves, fullinputprefix/readonlyfields/guards require exact inindependentperfieldchecker; macroscopicFP64onlyusestatedbudget.','extra_capture':'Fullreadonlyfixtureagainafterkernels plus70guardfields:7inputarraytypedguards(38scalarfields) andverification/detail2guardsper8cases(32).'}
(p/'independent_oracle.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
