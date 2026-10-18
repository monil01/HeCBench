"""CPU-only omitted-endpoint contribution, separate from unchanged primary contract."""
from pathlib import Path
import json,struct
p=Path(__file__).resolve().parent;source=(p/'study_oracle.py').read_text();ns={"__file__":str(p/"study_oracle.py")};exec(compile(source[:source.index('cache={};seed=1070')],str(p/'study_oracle.py'),'exec'),ns)
mp=ns['mp'];M=ns['M'];C=ns['C'];W=ns['W'];o=struct.unpack('<'+'d'*((p/'oracle_f64.bin').stat().st_size//8),(p/'oracle_f64.bin').read_bytes());cases=[]
for doppler,base in [(0,0),(1,9)]:
 row=o[base:base+9];E=M(row[1]);mat=int(row[2]);delta=[mp.mpf(0)]*4;terms=[]
 for j in range(ns['num'][mat]):
  nuc=ns['mats'][mat*ns['maxn']+j];window=min(int(E/(1/M(ns['nw'][nuc]))),ns['nw'][nuc]-1);w=ns['windows'][nuc*ns['maxw']+window];k=int(w[4])
  if k<int(w[3]):continue
  pole=ns['poles'][nuc*ns['maxp']+k];l=int(pole[8]);phi=M(ns['pseudo'][nuc*4+l])*mp.sqrt(E)
  if l==1:phi+=mp.atan(phi)
  elif l==2:phi-=mp.atan(3*phi/(3-phi*phi))
  elif l==3:phi-=mp.atan(phi*(15-phi*phi)/(15-6*phi*phi))
  sig=mp.mpc(mp.cos(2*phi),-mp.sin(2*phi));ea=C(*pole[:2]);rt=C(*pole[2:4]);ra=C(*pole[4:6]);rf=C(*pole[6:8]);term=W((E-ea)/2)if doppler else (1j/(ea-mp.sqrt(E)))/E
  micro=[(rt*term*sig).real,(ra*term).real,(rf*term).real];micro.append(micro[0]-micro[1]);conc=M(ns['concs'][mat*ns['maxn']+j]);contribution=[x*conc for x in micro]
  for q in range(4):delta[q]+=contribution[q]
  terms.append({'nuclide':nuc,'active_window':window,'assigned_interval_inclusive':[int(w[3]),k],'omitted_endpoint':k,'macro_delta':[str(x)for x in contribution]})
 cases.append({'lookup_index':0,'doppler':doppler,'energy':str(E),'material':mat,'primary_source_policy_macro':list(row[5:]),'counterfactual_add_inclusive_endpoint_macro_delta':[str(x)for x in delta],'nonzero_delta':any(x!=0 for x in delta),'per_nuclide':terms})
print(json.dumps({'scope':'IndependentCPUonlysourceinitializer-coverage witness. Addinginclusiveendpointchangesactualmacrovector. Thisisnotproofthatanexternalphysicalcontractrequiresit,nocorrectednativeclaimed,noacceptance/timingchange. Primarypreservessourceexclusiveend.','cases':cases},indent=2))
