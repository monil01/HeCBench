#!/usr/bin/env python3
"""Independent scalar Boolean-factorization replay; no CUDA/source helper imports."""
import json,math,struct,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
U=0xffffffff
def f(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def h(x):
 x=((x+0x7ed55d16)+(x<<12))&U;x=((x^0xc761c23c)^(x>>19))&U
 x=((x+0x165667b1)+(x<<5))&U;x=((x+0xd3a2646c)^(x<<9))&U
 x=((x+0xfd7046c5)+(x<<3))&U;return ((x^0xb55a4f09)^(x>>16))&U
def init(seed):
 w=seed or U
 for _ in range(8):x=h(w);y=h(x);z=h(y);w=h(z)
 return [x,y,z,w,0]
def draw(s):
 x,y,z,w,c=s;y^=(y<<5)&U;y^=y>>7;y^=(y<<22)&U;t=(z+w+c)&U
 x=(x+1411392427)&U;s[:]=[x,y,w,t&0x7fffffff,int(t>=0x80000000)]
 return (x+y+s[3])&U
def mask(s,dim,chance,depth):
 s=s.copy();r=f(f(draw(s))/f(U))
 if r<chance:
  m=(1<<dim)-1
  for _ in range(depth):m&=draw(s)
  return m
 return 1<<(draw(s)%dim)
def truth(c,i,j):return (c['C'][(i//32)*c['w']+j]>>(i%32))&1
def distance(c,A,B,weight=1):
 return sum((weight if truth(c,i,j) else 1) for i,a in enumerate(A) for j,b in enumerate(B) if bool(a&b)!=bool(truth(c,i,j)))
def replay(c):
 master=init(c['seed']);best=1e30;records=[];margin=1.;updates=0
 for run in range(c['runs']):
  seed=draw(master);runseed=draw(master);dim=c['dim'];threshold=c['threshold'];rd=int(-f(math.log2(threshold))+1)
  def factors(n,seed):
   out=[]
   for i in range(n):
    s=init((seed+i)&U);v=(1<<dim)-1
    if rd>=16:v=0
    else:
     for _ in range(rd):v&=draw(s)
    out.append(v)
   return out
  A=factors(c['h'],seed);B=factors(c['w'],(seed+c['h'])&U);s=init(runseed);temp=c['temp'];weight=float(c['weight'])
  total=distance(c,A,B,weight);polled=total;previous=total;stuck=0;iterations=0
  for iteration in range(1,c['iterations']+1):
   if not(polled>c['threshold_stop'] and temp>-1 and stuck<10000):break
   iterations=iteration
   for col in (False,True):
    n=c['w'] if col else c['h'];other=c['h'] if col else c['w'];target=B if col else A;opposite=A if col else B
    start=(draw(s)%n)//16*16;gseed=(draw(s)+iteration)&U;pad=((n+15)//16)*16
    for warp in range(((min(112,n)+15)//16)*16):
     at=(start+warp)%pad
     if at>=n:continue
     rng=init((gseed+warp)&U);new=target[at]^mask(rng,dim,c['chance'],2);old=target[at];delta=0
     for j,b in enumerate(opposite):
      t=truth(c,j,at) if col else truth(c,at,j);wt=weight if t else 1
      delta+=wt*((int(bool(new&b))^t)-(int(bool(old&b))^t))
     accept=delta<=0
     if delta>0 and temp>0:
      r=f(f(draw(rng))/f(U));prob=f(math.exp(f(f(-delta/other)/f(temp/10))))
      margin=min(margin,abs(r-prob));accept=r<prob
     if accept:target[at]=new;total+=delta;updates+=1
   if iteration%100==0:
    polled=total;stuck=stuck+100 if polled==previous else 0;previous=polled
   if iteration%c['reduce_step']==0:
    temp=f(temp*c['reduce_factor'])
    if weight>1:weight=f(weight*c['reduce_factor'])
    if weight<1:weight=1.
  dist=distance(c,A,B)
  records.append(f(f(dist/c['h'])/c['w']))
  if dist<best:best=dist;bestA=A.copy();bestB=B.copy()
 c['expected']=A+B+[float(dist)]+bestA+bestB+[float(best)]+records
 c['metropolis_min_margin']=margin;c['accepted_updates']=updates
 return c
def main():
 cases=[]
 for k,(hh,ww,dim,it,temp,weight) in enumerate([(17,33,5,7,0.,1),(65,97,13,105,.8,1),(257,529,20,21,0.,3)]):
  C=[0]*(((hh+31)//32)*ww);ones=0
  for i in range(hh):
   for j in range(ww):
    t=int((i*17+j*31+(i*j)%23)%11<3);ones+=t;C[(i//32)*ww+j]|=t<<(i%32)
  density=f(ones/(hh*ww));threshold=f(math.sqrt(1-math.pow(1-density,f(1/dim))))
  c=dict(h=hh,w=ww,dim=dim,iterations=it,temp=f(temp),weight=weight,seed=123+17*k,runs=2,chance=f(.1),reduce_step=20,reduce_factor=f(.98),threshold_stop=-1.,density=density,threshold=threshold,C=C)
  cases.append(replay(c))
 out=bytearray(struct.pack('<I',len(cases)))
 for c in cases:
  out+=struct.pack('<10I5f',c['h'],c['w'],c['dim'],c['iterations'],c['weight'],c['seed'],c['runs'],c['reduce_step'],len(c['C']),len(c['expected']),c['temp'],c['chance'],c['reduce_factor'],c['density'],c['threshold'])
  out+=struct.pack('<%dI'%len(c['C']),*c['C']);out+=struct.pack('<%dd'%len(c['expected']),*c['expected'])
 (P/'fixtures.bin').write_bytes(out)
 report=[{k:v for k,v in c.items() if k not in ('C','expected')} for c in cases]
 (P/'fixture_manifest.json').write_text(json.dumps(dict(generator='Independent scalar Python replay; full row/column candidate deltas and FastKISS modulo2^32; final factors/best/score, named singleactive slot sequential2runs',cases=report,sha256=hashlib.sha256(out).hexdigest(),metropolis_rule='Float32 uniform/probability; min margin reported, not MCMC mixing/optimality certificate'),indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
