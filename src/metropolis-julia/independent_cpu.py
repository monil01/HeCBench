# Independent direct global Ising neighbor oracle, not shared-tile source copy.
import math,struct
L=32;N=L**3;mask=(1<<64)-1;f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
idx=lambda x,y,z:(z%L)*L*L+(y%L)*L+(x%L)
def draw(st,inc):
 old=st;st=(old*6364136223846793005+inc)&mask;xo=(((old>>18)^old)>>27)&0xffffffff;rot=old>>59;u=((xo>>rot)|(xo<<((-rot)&31)))&0xffffffff;return st,f32(f32(u)*f32(2.3283064e-10))
def simulate(s,H,states,incs,h,B,sweeps):
 a=s.copy();st=states.copy();margin=1.;decisions=0
 for alt in [0,1]*sweeps:
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
    yy=y+((tx+tz+phase)&1);site=idx(x,yy,z);field=H[site];summ=sum(a[idx(x+dx,yy+dy,z+dz)] for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]);dh=a[site]*(summ+h*field);st[t],u=draw(st[t],incs[t]);prob=math.exp(dh*B);margin=min(margin,abs(u-prob));flip=(dh-f32(1e-11)<0)or(u<prob);updates.append((site,a[site]*(-1 if flip else 1)));decisions+=int(flip)
   for site,v in updates:a[site]=v
 return a,st,margin,decisions
