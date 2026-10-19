#!/usr/bin/env python3
"""Independent tensor D2Q9 BGK oracle; no native/GPU outputs used."""
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent
D=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
W=np.array([4/9,*([1/9]*4),*([1/36]*4)])
OP=[0,3,4,1,2,7,8,5,6]
CASES=[(128,8,1100,.2,.01,1.0,'obstacle_channel'),(256,16,1101,.2,.005,1.2,'periodic_islands'),(128,8,1001,.2,0.,.8,'equilibrium_control')]
def guard(a,g):return np.r_[np.full(16,g),a.ravel(),np.full(16,g)]
def solve(nx,ny,T,rho,accel,omega,kind):
 rho,accel,omega=map(lambda x:float(np.float32(x)),[rho,accel,omega]);da=float(np.float32(rho)*np.float32(accel))
 # C host initializer's operation ordering, then promote actual float32 inputs.
 weights=np.array([np.float32(np.float32(rho)*np.float32(4))/np.float32(9),*([np.float32(rho)/np.float32(9)]*4),*([np.float32(rho)/np.float32(36)]*4)],dtype='<f4')
 initial=np.broadcast_to(weights[:,None,None],(9,ny,nx)).copy();ob=np.zeros((ny,nx),dtype='<i4')
 if kind=='obstacle_channel':ob[1,:]=1;ob[ny-3,::7]=1;ob[ny-2,nx//2]=1
 if kind=='periodic_islands':
  for y in range(1,ny,4):ob[y,(np.arange(nx)+3*y)%23<3]=1
 f=initial.astype('f8');previous=None;vels=[];counts=[];mass0=float(f.sum());minpop=1.
 G=nx//128*ny
 for t in range(T):
  a=f.copy();row=ny-2;good=(ob[row]==0)&(f[3,row]>da/9)&(f[6,row]>da/36)&(f[7,row]>da/36)
  for q,sign in [(1,1),(5,1),(8,1),(3,-1),(6,-1),(7,-1)]:a[q,row,good]+=sign*da*(1/9 if q in (1,3) else 1/36)
  streamed=np.array([np.roll(a[q],(int(dy),int(dx)),axis=(0,1)) for q,(dx,dy) in enumerate(D)])
  dens=streamed.sum(axis=0);u=np.einsum('qa,qyx->ayx',D,streamed)/dens
  du=np.einsum('qa,ayx->qyx',D,u);u2=np.sum(u*u,axis=0)
  eq=W[:,None,None]*dens*(1+3*du+4.5*du*du-1.5*u2)
  after=streamed+omega*(eq-streamed);after[:,ob!=0]=streamed[OP][:,ob!=0]
  den=after.sum(axis=0);uv=np.einsum('qa,qyx->ayx',D,after)/den
  speed=np.sqrt(np.sum(uv*uv,axis=0));speed[ob!=0]=0
  vels.append(speed.reshape(ny,nx//128,128).sum(axis=2).ravel());counts.append((ob==0).reshape(ny,nx//128,128).sum(axis=2).ravel());previous=f;f=after
  minpop=min(minpop,float(f.min()))
 ps=np.array(vels);pc=np.array(counts,dtype='<i4')
 assert np.isfinite(f).all() and minpop>0 and abs(f.sum()/mass0-1)<1e-9
 N=nx*ny;f0=initial.ravel();zero=np.zeros(9*N,dtype='<f4')
 def prefix():
  a=[nx,ny,T,rho,accel,omega,da]
  for q in range(9):a.extend(guard(f0[q*N:(q+1)*N],-99))
  for q in range(9):a.extend(guard(zero[q*N:(q+1)*N],-99))
  a.extend(guard(ob,-12345));a.extend(guard(np.zeros(G*T),-99));a.extend(guard(np.zeros(G*T),-12345));return a
 a=prefix();prefixcount=len(a)
 for q in range(9):a.extend(guard(f[q],-99))
 for q in range(9):a.extend(guard(previous[q],-99))
 a.extend(guard(ps,-99));a.extend(guard(pc,-12345))
 return np.array(a,dtype='<f8'),initial,ob,{'nx':nx,'ny':ny,'steps':T,'density':rho,'accel':accel,'omega':omega,'densityaccel':da,'kind':kind,'prefix_count':prefixcount,'count':len(a),'mass_initial':mass0,'mass_final_F64':float(f.sum()),'minimum_population_F64':minpop,'source_timed_launches':T-999,'source_printed_divisor':T-1000}
def main():
 rows=[];oracle=[];prefixes=[];tails=[]
 for k,c in enumerate(CASES):
  out,f,ob,d=solve(*c);f.tofile(P/f'initial{k}_f32.bin');ob.tofile(P/f'obstacles{k}_i32.bin');n=d['prefix_count'];prefixes.extend(out[:n]);tails.extend(out[n:]);rows.append(d)
 out=np.array(prefixes+tails,dtype='<f8');out.tofile(P/'oracle_f64.bin')
 d={'cases':rows,'prefix_count':len(prefixes),'count':len(out),'atol':2e-5,'rtol':2e-4,'oracle':'Independent Float64 tensor moments/equilibrium, periodic np.roll streaming, source-defined positive force, obstacle reversal; full final and penultimate distributions and every group velocity/count at every timestep. Exact actual inputs/guards/counts required separately.','source_timing':'Original loop starts its timer before t=999; preserve T-999 launches after999 evolves, without dividing by erroneous source T-1000 label. Reset all state per trial; complete-context fences; allocation/copy/oracle outside.'}
 d['files']={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in list(P.glob('initial*_f32.bin'))+list(P.glob('obstacles*_i32.bin'))+[P/'oracle_f64.bin']}
 (P/'study_plan.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'cases':len(rows),'prefix':len(prefixes),'count':len(out)}))
if __name__=='__main__':main()
