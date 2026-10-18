"""Independent array-based staggered shallow-water evaluator; NumPy Float64.
The boundary/update ordering is the original EasyWave observable contract, not
an independent proof that its corner conditions are physically complete.
"""
from pathlib import Path
import numpy as np,json,hashlib,sys
P=Path(__file__).resolve().parent;NX=23;NY=19;STEPS=8
D,H,HM,M,N,R1,R2,R3,R4,R5,T,TOPO=range(12)
def advance(a,b,r6,C,step):
 trace=[];il,jl,ih,jh=b
 # Interior finite-volume mass balance, vectorized independently from CUDA indexing.
 ix=np.arange(il-1,ih);jy=np.arange(jl-1,jh);z=np.ix_(ix,jy);old=a.copy()
 flux=a[z+(M,)]-a[np.ix_(ix-1,jy)+(M,)]+a[z+(N,)]*r6[jy+1]-a[np.ix_(ix,jy-1)+(N,)]*r6[jy]
 wet=a[z+(D,)]!=0;value=a[z+(H,)]-a[z+(R1,)]*flux;value=np.where(abs(value)<1e-6,0,value)
 a[z+(H,)]=np.where(wet,value,a[z+(H,)])
 a[z+(HM,)]=np.where(wet,np.maximum(a[z+(HM,)],a[z+(H,)]),a[z+(HM,)])
 arrival=wet&(a[z+(T,)]<0)&(abs(old[z+(H,)]-old[z+(R1,)]*flux)>1e-4)
 a[z+(T,)]=np.where(arrival,step,a[z+(T,)])
 trace.append(a.copy())
 # Boundary radiation magnitudes. Explicit ordered endpoint assignments preserve source policy.
 def rad(i,j,u,v,c,negative):a[i,j,H]=(-1 if negative else 1)*np.hypot(u,v)*c
 if jl<=2:
  for i in range(1,NX-1):rad(i,0,a[i,0,N],.5*(a[i,0,M]+a[i-1,0,M]),C[0][i+1],a[i,0,N]>0)
 if il<=2:
  for j in range(1,NY-1):rad(0,j,a[0,j,M],.5*(a[0,j,N]+a[0,j-1,N]),C[1][j+1],a[0,j,M]>0)
 if jh>=NY-1:
  for i in range(1,NX-1):rad(i,NY-1,a[i,NY-2,N],.5*(a[i,NY-1,M]+a[i,NY-2,M]),C[2][i+1],a[i,NY-2,N]<0)
 if ih>=NX-1:
  for j in range(1,NY-1):rad(NX-1,j,a[NX-2,j,M],.5*(a[NX-1,j,N]+a[NX-1,j-1,N]),C[3][j+1],a[NX-2,j,M]<0)
 if jl<=2:
  rad(0,0,a[0,0,M],a[0,0,N],C[0][1],a[0,0,N]>0)
  rad(NX-1,0,a[NX-2,0,M],a[NX-1,0,N],C[0][NX],a[NX-1,0,N]>0)
 if jl>=NY-1: # Original Jmin predicate, not silently repaired to Jmax.
  rad(0,NY-1,a[0,NY-1,M],a[0,NY-2,N],C[2][1],a[0,NY-2,N]<0)
  rad(NX-1,NY-1,a[NX-2,NY-1,M],a[NX-1,NY-2,N],C[2][NX],a[NX-1,NY-2,N]<0)
 trace.append(a.copy())
 wetx=(a[z+(D,)]*a[np.ix_(ix+1,jy)+(D,)])!=0;wety=(a[z+(D,)]*a[np.ix_(ix,jy+1)+(D,)])!=0
 a[z+(M,)]=np.where(wetx,a[z+(M,)]-a[z+(R2,)]*(a[np.ix_(ix+1,jy)+(H,)]-a[z+(H,)]),a[z+(M,)])
 a[z+(N,)]=np.where(wety,a[z+(N,)]-a[z+(R4,)]*(a[np.ix_(ix,jy+1)+(H,)]-a[z+(H,)]),a[z+(N,)])
 trace.append(a.copy())
 def mx(i,j):a[i,j,M]-=a[i,j,R2]*(a[i+1,j,H]-a[i,j,H])
 def my(i,j):a[i,j,N]-=a[i,j,R4]*(a[i,j+1,H]-a[i,j,H])
 if jl<=2:
  for i in range(NX-1):mx(i,0)
 if il<=2:
  for j in range(NY):mx(0,j)
 if jh>=NY-1:
  for i in range(NX-1):mx(i,NY-1)
 if il<=2:
  for j in range(NY-1):my(0,j)
 if jl<=2:
  for i in range(NX):my(i,0)
 if ih>=NX-1:
  for j in range(NY-1):my(NX-1,j)
 # Area growth uses the current stage's height and sequentially updated extents.
 if il>2 and np.any(abs(a[il+1,jl-1:jh,H])>1e-4):il=max(2,il-1)
 if ih<NX-1 and np.any(abs(a[ih-3,jl-1:jh,H])>1e-4):ih=min(NX-1,ih+1)
 if jl>2 and np.any(abs(a[il-1:ih,jl+1,H])>1e-4):jl=max(2,jl-1)
 if jh<NY-1 and np.any(abs(a[il-1:ih,jh-3,H])>1e-4):jh=min(NY-1,jh+1)
 trace.append(a.copy());return trace,[il,jl,ih,jh]
def main():
 files={};cases=[];physics=[]
 def write(name,a):f=P/name;np.asarray(a).tofile(f);files[name]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
 for case in (1,2):
  a=np.zeros((NX,NY,12),dtype='<f4');a[:,:,D]=4;a[:,:,T]=-1;a[:,:,TOPO]=-4
  a[10:13,8:11,D]=0;a[10:13,8:11,TOPO]=1
  wet=a[:,:,D]!=0;a[:,:,R1]=np.where(wet,1/32,0)
  a[:-1,:,R2]=np.where(wet[:-1,:]&wet[1:,:],1/8,0);a[:,:-1,R4]=np.where(wet[:,:-1]&wet[:,1:],1/8,0)
  if case==1:
   for i in range(4,19):
    for j in range(4,15):
     if wet[i,j]:a[i,j,H]=((i*7+j*3)%11-5)/1024
   a[:,:,HM]=np.maximum(a[:,:,H],0)
  bounds=[5,5,19,15] if case==1 else [2,2,NX-1,NY-1]
  r6=np.ones(NY+1,dtype='<f4');C=[np.full(n+1,.5,dtype='<f4') for n in (NX,NY,NX,NY)]
  fixture=np.concatenate([a.ravel(),r6,*C]).astype('<f4');write(f'study_case{case}_f32.bin',fixture)
  current=a.astype(np.float64);b=bounds.copy();traces=[];bt=[];mass_res=[];moment_res=[];hm_previous=current[:,:,HM].copy()
  for step in range(STEPS):
   before=current.copy();oldb=b.copy();tr,b=advance(current,b,r6.astype(float),[c.astype(float) for c in C],step)
   # Separate physical/discrete checks: exact resting equilibrium and source stage invariants.
   assert np.all(current[:,:,HM]>=hm_previous);hm_previous=current[:,:,HM].copy();assert b[0]<=oldb[0] and b[1]<=oldb[1] and b[2]>=oldb[2] and b[3]>=oldb[3]
   if case==2:assert np.array_equal(current,a.astype(float))
   # Interior mass flux balance residual, excluding deliberate zero-threshold truncation.
   il,jl,ih,jh=oldb;v=before[il-1:ih,jl-1:jh];after=tr[0][il-1:ih,jl-1:jh];div=(v[:,:,M]-before[il-2:ih-1,jl-1:jh,M]+v[:,:,N]-before[il-1:ih,jl-2:jh-1,N]);res=(after[:,:,H]-v[:,:,H])+v[:,:,R1]*div;mask=(v[:,:,D]!=0)&(after[:,:,H]!=0);mass_res.append(float(np.max(abs(res[mask]))) if np.any(mask) else 0.)
   for stage in tr:traces.extend(stage.ravel())
   bt.extend(b)
  write(f'oracle_case{case}_f64.bin',np.array(list(current.ravel())+traces+bt,dtype='<f8'))
  cases.append(dict(case=case,bounds=bounds,final_bounds=b,input_count=len(fixture),trace_count=STEPS*4*NX*NY*12,bounds_trace_count=STEPS*4,oracle_count=current.size+len(traces)+len(bt),timed_final_count=current.size,resting_equilibrium=case==2))
  physics.append(dict(case=case,interior_discrete_mass_residual_max=max(mass_res),max_height_non_decreasing=True,bounds_monotone=True,resting_equilibrium_exact=case==2,physical_scope='Nondimensional linear shallow water: g1,D4,dx=dy1,dt1/32,R6=1 Cartesian metric; no earthquake source/inundation validation.'))
 out=dict(cases=cases,files=files,physics_controls=physics,oracle='Independent Float64 vectorized staggered mass/momentum update with hypot-based open-boundary magnitudes; serial boundary/order/area policy explicitly follows source, not independent validation of all corner physics.',generator=dict(path='src/easyWave-julia/generate_oracle.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),numpy=np.__version__,python=sys.version,nx=NX,ny=NY,steps=STEPS,input_prefix_count=2*len(fixture)+2*13,output_count=sum(c['oracle_count'] for c in cases),guard_count=2*2*10*32,atol=1e-7,rtol=1e-5)
 (P/'fixture_provenance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
