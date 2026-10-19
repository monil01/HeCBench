"""Independent F64 mathematical semblance; parse actual SU bytes independently."""
from pathlib import Path
from array import array
import struct,math,json,hashlib,sys
P=Path(__file__).resolve().parent;assert sys.byteorder=='little'
x=array('d');x.frombytes((P/'fixture.bin').read_bytes());assert x[0]==2;k=1;gold=array('d');review=[]
for ci in range(2):
 nc,ns,ncdps,nt,maxtr=map(int,x[k:k+5]);c0,inc,dt,idt,tau,w=x[k+5:k+11];tau=int(tau);w=int(w);k+=11
 counts=list(map(int,x[k:k+ncdps]));k+=ncdps;scalco,gx,gy,sx,sy=[x[k+q*nt:k+(q+1)*nt]for q in range(5)];k+=5*nt;samples=x[k:k+nt*ns];k+=nt*ns
 raw=(P/f'synthetic{ci}.su').read_bytes();size=240+4*ns;assert len(raw)%size==0;parsed={};rejected=0
 for i in range(len(raw)//size):
  offset=i*size;head=raw[offset:offset+240];cdp=struct.unpack_from('<i',head,20)[0];s=struct.unpack_from('<h',head,70)[0];xx,yy,xxx,yyy=struct.unpack_from('<iiii',head,72);n0,dt0=struct.unpack_from('<HH',head,114);assert n0==ns and dt0==2000
  scale=1 if s==0 else s if s>0 else 1/s
  if .25*((xxx-xx)**2+(yyy-yy)**2)*scale**2>600**2:rejected+=1;continue
  vals=struct.unpack_from('<'+'f'*ns,raw,offset+240);parsed.setdefault(cdp,[]).append((s,xxx,yyy,xx,yy,vals))
 actual=[q for c in sorted(parsed)for q in parsed[c]];assert len(actual)==nt and rejected==1;cum=0
 for ii,c in enumerate(sorted(parsed)):cum+=len(parsed[c]);assert counts[ii]==cum
 for i,q in enumerate(actual):
  assert tuple(float(v)for v in q[:5])==(scalco[i],gx[i],gy[i],sx[i],sy[i]);assert tuple(samples[i*ns:(i+1)*ns])==q[5]
 c=[c0+inc*i for i in range(nc)];h=[]
 for i in range(nt):
  scale=1 if abs(scalco[i])<1e-13 else scalco[i] if scalco[i]>0 else 1/scalco[i]
  h.append(((gx[i]-sx[i])**2+(gy[i]-sy[i])**2)*scale**2/(4*1e6))
 gold.extend(c);gold.extend(h);start=0;groups=[];boundmargin=1.;denmargin=1.;peakcertificate=None
 for group,stop in enumerate(counts):
  sem=[];stack=[];valid=[]
  for t0 in range(ns):
   for curvature in c:
    # Independently form an interpolated trace-window matrix, then column sums.
    matrix=[];bad=0
    for tr in range(start,stop):
     travel=math.sqrt((dt*t0)**2+curvature*h[tr])*idt;it=int(travel);fraction=travel-it
     # Branch boundaries, not interpolation-grid crossings, govern validity.
     boundmargin=min(boundmargin,abs(travel-tau),abs(travel-(ns-tau-1)))
     if it-tau<0 or it+tau+1>=ns:bad+=1;continue
     base=tr*ns+it-tau
     matrix.append([samples[base+j]*(1-fraction)+samples[base+j+1]*fraction for j in range(w)])
    m=len(matrix);den=sum(v*v for row in matrix for v in row)
    if den>0:denmargin=min(denmargin,abs(den-1e-13)/max(den,1e-13))
    if den>1e-13 and m>0 and w>0 and bad<2:
     cols=[sum(row[j]for row in matrix)for j in range(w)];score=sum(v*v for v in cols)/(den*m);mean=sum(cols)/(w*m)
     assert -1e-12<=score<=1+1e-12;sem.append(score);stack.append(mean);valid.append(True)
    else:sem.append(-1.);stack.append(-1.);valid.append(False)
  ctr=[];strength=[];stk=[];mingap=1.
  for t0 in range(ns):
   vals=sem[t0*nc:(t0+1)*nc];best=max([0.]+vals);ix=vals.index(best)if best>0 else -1;ctr.append(ix);strength.append(best);stk.append(stack[t0*nc+ix]if ix>=0 else 0.)
   if best>0:
    gap=best-max(v for ii,v in enumerate(vals)if ii!=ix);mingap=min(mingap,gap)
  if group==0:
   at=100 if ci==0 else 300;peakcertificate={'t0_sample':at,'selected_c_index':ctr[at],'expected_injected_c_index':nc//2,'selected_semblance':strength[at],'next_best_gap':strength[at]-max(sem[at*nc+i]for i in range(nc)if i!=ctr[at])};assert ctr[at]==nc//2
  if group==1:
   assert all(abs(v-1)<1e-12 for v,ok in zip(sem,valid)if ok)
  if group==2:assert all(v==-1 for v in sem)and all(v==-1 for v in ctr)
  gold.extend(sem);gold.extend(stack);gold.extend(ctr);gold.extend(strength);gold.extend(stk)
  groups.append({'group':group,'traces':stop-start,'valid_scan_values':sum(valid),'invalid_scan_values':len(valid)-sum(valid),'minimum_positive_argmax_gap_including_ties':mingap})
  start=stop
 review.append({'case':ci,'native_actual_SUlinearization_exactly_matches_independent_SUparse':True,'rejected_aperture_traces':rejected,'ns':ns,'nc':nc,'trace_counts':counts,'window':w,'minimum_validity_branch_distance_samples':boundmargin,'minimum_relative_energy_threshold_distance':denmargin,'injected_event_curvature_certificate':peakcertificate,'groups':groups})
assert k==len(x);(P/'oracle.bin').write_bytes(gold.tobytes());(P/'oracle_review.json').write_text(json.dumps({'cases':review,'oracle_values':len(gold),'oracle_sha256':hashlib.sha256(gold.tobytes()).hexdigest(),'method':'IndependentFloat64interpolatedtrace-window matrix andCauchycoherence/stack/firststrictpositiveargmax; exactoriginalSUCPUlinearization independentlychecked. Sourcec0/inc/dt/idtFloat32arguments expanded, not originalCPUchecker.','physical_scope':'Knownsynthetichyperbolicmoveoutcurvaturecontrols; no measuredsubsurfacevelocity/applicationaccuracy ororiginal81MBDVCdatasetclaim.'},indent=2)+'\n');print(json.dumps(review,indent=2))
