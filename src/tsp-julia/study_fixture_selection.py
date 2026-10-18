from pathlib import Path
import random,math,struct,json,time
R=Path('/home/imo/HeCBench_julia');p=R/'src/tsp-julia';p.mkdir(exist_ok=True)
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def threads(n):
 best=0;bt=4
 for t in range(1,min(n-2,256)+1):
  blocks=min(32768//(4*t+1536),16);thr=((t+31)//32)*32
  while blocks*thr>2048:blocks-=1
  if t*blocks>best:best=t*blocks;bt=t
 return bt

def replay(points,restart,nt):
 n=len(points);route=list(range(n));seed=restart;draws=[]
 for i in range(1,n):
  seed=(26757677*seed+1)%2147483648;r=f32(f32(seed)/f32(2147483648));j=int(f32(r*f32(n-1)))+1
  if j>=n:return None,{'rng_out_of_range':True,'i':i,'seed':seed}
  route[i],route[j]=route[j],route[i];draws.append({'seed':seed,'j':j})
 ds=[[int(f32(math.sqrt(f32((x-u)*(x-u)+(y-v)*(y-v)))))for u,v in points]for x,y in points]
 steps=[]
 for iteration in range(10000):
  edges=[ds[route[i]][route[(i+1)%n]]for i in range(n)];bests=[]
  for tid in range(nt):
   best=0;mi=1;mj=0
   for i in range(tid,n-2,nt):
    for j in range(n-1,i+1,-1):
     change=ds[route[i]][route[j]]+ds[route[i+1]][route[(j+1)%n]]-edges[i]-edges[j]
     if change<best:best=change;mi=i;mj=j
   bests.append((best,mi,mj))
  low=min(x[0]for x in bests);winners=[i for i,x in enumerate(bests)if x[0]==low]
  if low<0 and len(winners)!=1:return None,{'iteration':iteration,'best_change':low,'negative_winner_lanes':winners}
  steps.append({'iteration':iteration,'lane_best_changes':[x[0]for x in bests],'winner_lanes':winners,'chosen_move':list(bests[winners[0]][1:])if low<0 else None,'route_before':route.copy(),'tour_length_before':sum(edges)})
  if low==0:return {'restart':restart,'threads':nt,'draws':draws,'steps':steps,'route':route,'edges':edges,'length':sum(edges),'climbs':iteration+1},None
  _,i,j=bests[winners[0]];route[i+1:j+1]=reversed(route[i+1:j+1])
 raise RuntimeError('iterations')
allcases=[];rejections=[]
for n in [100,131]:
 nt=threads(n);found=False
 for coordseed in range(1000):
  rng=random.Random(2026101800+n*1000+coordseed);base=[]
  while len(base)<n:
   q=(rng.randrange(1024),rng.randrange(1024))
   if q not in base:base.append(q)
  points=[(x*256,y*256)for x,y in base];out=[]
  for restart in range(3):
   a,b=replay(points,restart,nt)
   if b:rejections.append({'n':n,'coordinate_seed_offset':coordseed,'restart':restart,'reason':b});break
   out.append(a)
  else:
   allcases.append({'n':n,'restarts':3,'threads':nt,'coordinate_seed':2026101800+n*1000+coordseed,'points':points,'results':out});print('selected',n,nt,coordseed,[a['climbs']for a in out],flush=True);found=True;break
 if not found:raise RuntimeError(('no unique fixture',n))
(p/'deterministic_fixture_selection.json').write_text(json.dumps({'cases':allcases,'rejected_CPU_coordinate_candidates':rejections,'selection_rule':'BeforeGPU: firstcoordinateSeed withuniqueStrictlyNegativeBestLaneeveryactualrestart/iteration, terminalzero tiesretained(nonimprovingno-op). Allactualcandidate lanes recorded. SourceLCG Float32multiplication+permutation preserved.','distance_control':'Integerbase0..1023 times256 =>dx/dy/dx²+dy² exactlyFloat32; Float32sqrt floor sourcepolicy, notTSPLIBnearestnorm/globaloptimum.'},indent=2)+'\n')
