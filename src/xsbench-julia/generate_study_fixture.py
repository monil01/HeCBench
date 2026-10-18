from pathlib import Path
from fractions import Fraction as F
import json,struct,bisect
p=Path(__file__).resolve().parent;raw=(p/'materials.bin').read_bytes();ma=struct.unpack('<'+str(len(raw)//4)+'i',raw);assert ma[:2]==(12,34);num=list(ma[2:14]);mats=list(ma[14:]);I=68;M=34;pre=[6];gold=[];kind=bytearray();cases=[];MASK=(1<<63)-1
D=[F(x)for x in ['.140','.052','.275','.134','.154','.064','.066','.055','.008','.015','.025','.013']];LCG=lambda s:(2806196910506780709*s+1)&MASK
samples=[];seed=1070;minroll=1
for i in range(1025):
 seed=LCG(seed);energy=float(seed)/2**63;seed=LCG(seed);roll=float(seed)/2**63;mat=0
 for m in range(12):
  threshold=sum(D[j]for j in range(m,0,-1));minroll=min(minroll,abs(float(threshold)-roll))
  if F(roll)<threshold:mat=m;break
 samples.append((F(energy),mat))
margin=1e100;labelset=set()
for G,L,reps,bins in [(17,257,2,8),(33,1025,3,32)]:
 grids=[];energies=[]
 for nuc in range(I):
  es=[F(0)if j==0 else F(1)if j==G-1 else F(j,G-1)+F(nuc%5-2,2048)for j in range(G)];energies.append(es)
  for e in es:
   elastic=F(nuc%11+2,128)*(1+e);absorp=F(nuc%7+2,16)*(F(3,2)-e/2);fission=absorp*(F(1,8)+F(3,4)*e);grids.extend(map(float,[e,elastic+absorp,elastic,absorp,fission,F(5,2)*fission]))
 concs=[float(F((m+1)*(j+3)%19+1,32))if j<num[m]else 0. for m in range(12)for j in range(M)];phys=[];labels=[]
 for e,mat in samples[:L]:
  macro=[F(0)]*5
  for j in range(num[mat]):
   nuc=mats[mat*M+j];es=energies[nuc];lo=max(0,min(G-2,bisect.bisect_right(es,e)-1));ratio=(e-es[lo])/(es[lo+1]-es[lo]);weight=F(concs[mat*M+j])
   for k in range(5):macro[k]+=weight*((1-ratio)*F(grids[6*(nuc*G+lo)+k+1])+ratio*F(grids[6*(nuc*G+lo+1)+k+1]))
  best=max(range(5),key=lambda k:macro[k]);ordered=sorted(macro,reverse=True);margin=min(margin,float(ordered[0]-ordered[1]));labels.append(best+1);labelset.add(best+1);phys.extend([float(e),float(mat),*map(float,macro)])
 for gridtype in [0,1,2]:
  union=sorted(float(e)for es in energies for e in es)if gridtype==0 else[0.];idx=[]
  if gridtype==0:
   for u in union:
    for es in energies:idx.append(max(0,min(G-2,bisect.bisect_right(es,F(u))-1)))
  elif gridtype==2:
   for b in range(bins):
    for es in energies:idx.append(max(0,min(G-2,bisect.bisect_right(es,F(b,bins))-1)))
  else:idx=[0]
  pre.extend([gridtype,I,G,L,reps,bins,M,len(union),len(idx)]);arrays=[(num,1),(concs,1),(mats,1),(grids,6),(union,1),(idx,1)]
  for a,w in arrays:pre.extend(a)
  def append(values,width=1,exact=True):gold.extend([16384]*(64*width)+list(values)+[16384]*(64*width));kind.extend(bytes([1])*(64*width));kind.extend(bytes([1 if exact else 0])*len(values));kind.extend(bytes([1])*(64*width))
  append(labels);start=len(gold);append(phys,exact=False)
  for i in range(L):kind[start+64+7*i]=1;kind[start+64+7*i+1]=1
  for a,w in arrays:append(a,w)
  cases.append(dict(grid_type=gridtype,isotopes=I,gridpoints=G,lookups=L,repeats=reps,hash_bins=bins,max_nuclides=M,union_count=len(union),index_count=len(idx),labels=sorted(set(labels))))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'oracle.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));(p/'kinds.bin').write_bytes(kind);(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_count=len(gold),artifact_count=len(pre)+len(gold),labels=sorted(labelset),minimum_exact_macro_winner_margin=margin,minimum_material_roll_threshold_margin=minroll,oracle='Sequentialbigintmod2^63LCG independentfromfastforward; exactFractionbracket/interpolation/materialconcentrationweightedfivechannels usingactualF64tablebytes; firststrictmaximum.',source_materials='Unchangedoriginalload_num_nucs(68)/load_mats(68)compilerCPUfixture; unusedslotsinitialized-777/0.',physics_constraints='Positivechannels; total=elastic+absorption, fission<=absorption, nu_fission=2.5*fission preservedbyconvexinterpolation/positivesums. Controlledsynthetictablesnotmeasurednucleardata/fulltransport.',diagnostic='Primarysourcekernelstoresonlyargmaxlabel; separateuntimedobserverstoresactualenergy/material/all5macrosusingunchangednativehelpers. Privateprimarymacroarraysnotdirectlycaptured.'),indent=2)+'\n');print(len(pre),len(gold),sorted(labelset),margin,minroll)
