from pathlib import Path
import math,struct,json
p=Path(__file__).resolve().parent;L=32;N=L**3;mask=(1<<64)-1;f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
idx=lambda x,y,z:(z%L)*L*L+(y%L)*L+(x%L)
H=[1 if (x*7+y*3+z)%5<2 else -1 for z in range(L) for y in range(L) for x in range(L)];s=[1]*N;states=[((i+1)*0x9e3779b97f4a7c15)&mask for i in range(N//4)];incs=[2*i+1 for i in range(N//4)]
def draw(st,inc):
 old=st;st=(old*6364136223846793005+inc)&mask;xo=(((old>>18)^old)>>27)&0xffffffff;rot=old>>59;u=((xo>>rot)|(xo<<((-rot)&31)))&0xffffffff;return st,f32(f32(u)*f32(2.3283064e-10))
def simulate(correct):
 a=s.copy();st=states.copy();margin=1.;decisions=0
 for alt in [0]:
  records=[]
  for bz in range(L//4):
   for by in range(L//16):
    for bx in range(L//16):
     oy=(2*by+((bx+bz+alt)&1))*8
     for tz in range(4):
      for ty in range(4):
       for tx in range(16):records.append((bx*16+tx,oy+2*ty,bz*4+tz,tx,tz,(bz*4+tz)*L*L//4+(by*4+ty)*L+bx*16+tx))
  for phase in [0,1]:
   updates=[]
   for x,y,z,tx,tz,t in records:
    yy=y+((tx+tz+phase)&1);site=idx(x,yy,z);field=H[site] if correct else H[idx(x,y+phase,z)];summ=sum(a[idx(x+dx,yy+dy,z+dz)] for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]);dh=a[site]*(summ+.5*field);st[t],u=draw(st[t],incs[t]);prob=math.exp(dh*-.5);margin=min(margin,abs(u-prob));flip=(dh-f32(1e-11)<0)or(u<prob);updates.append((site,a[site]*(-1 if flip else 1)));decisions+=int(flip)
   for site,v in updates:a[site]=v
 return a,st,margin,decisions
def encode(a,st):
 out=list(a)
 for arr in [st,incs]:
  for x in arr:out.extend([x&0xffffffff,x>>32])
 return out
raw=s+H
for arr in [states,incs]:
 for x in arr:raw.extend([x&0xffffffff,x>>32])
for name,values in [('witness_input.bin',raw),('witness_source_gold.bin',encode(*simulate(False)[:2])),('witness_ising_gold.bin',encode(*simulate(True)[:2]))]:p.joinpath(name).write_bytes(struct.pack('<'+'d'*len(values),*values))
bad=simulate(False);good=simulate(True);w=dict(L=L,N=N,alt=0,h=.5,B=-.5,source_block=[16,4,4],source_grid=[2,2,8],source_vs_Ising_spin_mismatches=sum(a!=b for a,b in zip(bad[0],good[0])),source_min_acceptance_margin=bad[2],correct_min_acceptance_margin=good[2],source_flips=bad[3],correct_flips=good[3],local_witness={'configuration':'all neighbors +1, central spin +1, local parity odd so white target yodd; H[yeven]=+1,H[yodd]=-1,h=.5,B=-.5','source_dh':6.5,'Ising_half_delta_E':5.5,'source_acceptance':math.exp(-3.25),'Ising_acceptance':math.exp(-2.75),'u_control':.05,'different_decision':True},qualification='Independent local energy-flip difference proves field mismatch for nonuniformH. Direct global neighbor CPU oracle uses legal blockcheckerboard schedule, source behavior and physically indexed Ising variants kept separately. NativeGPU observed verdict pending; no source defect conclusion solely from hazard. RawPCGstates/inc frozen namedinputs, no complete seededCLI/exchangetemperature/adaptation validation.')
p.joinpath('witness_description.json').write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w))
