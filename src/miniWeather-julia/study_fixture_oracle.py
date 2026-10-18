from pathlib import Path
import json,math,struct,decimal,hashlib
p=Path(__file__).resolve().parent
C0=27.5629410929725921310572974482;GAMMA=1.40027894002789400278940027894;g=9.8;CP=1004.;CV=717.;RD=287.;P0=1e5
Q=[.112701665379258311482073460022,.5,.887298334620741688517926539980];W=[.277777777777777777777777777779,.444444444444444444444444444444,.277777777777777777777777777779]
decimal.getcontext().prec=50
DC=decimal.Decimal.from_float(C0);DG=decimal.Decimal.from_float(GAMMA)
def pressure(rt):return float(DC*decimal.Decimal.from_float(rt)**DG)
def hydro(z):
 ex=1-g*z/(CP*300.);pr=P0*ex**(CP/RD);rt=(pr/C0)**(1/GAMMA);return rt/300.,rt

def make(nx,nz,steps,kind):
 dx=20000./nx;dz=10000./nz;dt=min(dx,dz)/450*1.5;S=[0.]*(4*(nx+4)*(nz+4));idx=lambda l,k,i:(l*(nz+4)+k)*(nx+4)+i
 if kind!='rest':
  for k in range(nz+4):
   for i in range(nx+4):
    for kk in range(3):
     for ii in range(3):
      x=(i-2+Q[ii])*dx;z=(k-2+Q[kk])*dz;rho,rt=hydro(z);dist=math.hypot((x-10000)/2000,(z-2000)/2000)*math.pi/2;th=3*math.cos(dist)**2 if dist<=math.pi/2 else 0;S[idx(3,k,i)]+=rho*th*W[ii]*W[kk]
      if kind=='moving_thermal':S[idx(1,k,i)]+=rho*(.1*math.sin(2*math.pi*x/20000))*W[ii]*W[kk]
 hy0=[];hy1=[]
 for k in range(nz+4):
  rho,rt=hydro((k-1.5)*dz);hy0.append(math.fsum(rho*w for w in W));hy1.append(math.fsum(rt*w for w in W))
 hi0=[];hi1=[];hip=[]
 for k in range(nz+1):rho,rt=hydro(k*dz);hi0.append(rho);hi1.append(rt);hip.append(C0*rt**GAMMA)
 return {'nx':nx,'nz':nz,'steps':steps,'dx':dx,'dz':dz,'dt':dt,'schedule':[dt]*(steps-1)+[dt*.5], 'initial':S,'hy':[hy0,hy1,hi0,hi1,hip],'kind':kind}
def solve(c):
 nx=c['nx'];nz=c['nz'];dx=c['dx'];dz=c['dz'];rho,rt,ri,rti,pi=c['hy'];state=c['initial'].copy();tmp=state.copy();flux=[0.]*(4*(nx+1)*(nz+1));tend=[0.]*(4*nx*nz);sl=sr=rl=rr=[0.]*(8*nz);idx=lambda l,k,i:(l*(nz+4)+k)*(nx+4)+i;fx=lambda l,k,i:(l*(nz+1)+k)*(nx+1)+i;tx=lambda l,k,i:(l*nz+k)*nx+i
 def diagnostics(s):
  mass=[];energy=[]
  for k in range(nz):
   for i in range(nx):
    r=s[idx(0,k+2,i+2)]+rho[k+2];u=s[idx(1,k+2,i+2)]/r;w=s[idx(2,k+2,i+2)]/r;th=(s[idx(3,k+2,i+2)]+rt[k+2])/r;pr=pressure(r*th);temperature=th/(P0/pr)**(RD/CP);mass.append(r*dx*dz);energy.append((r*(u*u+w*w)+r*CV*temperature)*dx*dz)
  return [math.fsum(mass),math.fsum(energy)]
 initial_diagnostics=diagnostics(state)
 def stage(forcing,out,subdt,axis,fulldt):
  nonlocal sl,sr,rl,rr
  if axis==0:
   # Independent periodiccopy expressed fromphysicalleft/rightinterior; savedsourcepackbuffers tracked separately.
   sl=[forcing[idx(l,k+2,2+s)]for l in range(4)for k in range(nz)for s in range(2)];sr=[forcing[idx(l,k+2,nx+s)]for l in range(4)for k in range(nz)for s in range(2)];rl=sr.copy();rr=sl.copy()
   for l in range(4):
    for k in range(nz):
     for ss in range(2):forcing[idx(l,k+2,ss)]=rl[(l*nz+k)*2+ss];forcing[idx(l,k+2,nx+2+ss)]=rr[(l*nz+k)*2+ss]
  else:
   for l in range(4):
    for i in range(nx+4):
     for kk,source in [(0,2),(1,2),(nz+2,nz+1),(nz+3,nz+1)]:forcing[idx(l,kk,i)]=0. if l==2 else forcing[idx(l,source,i)]
  hv=-.25*(dx if axis==0 else dz)/(16*fulldt)
  for k in range(nz+(axis==1)):
   for i in range(nx+(axis==0)):
    vals=[];third=[]
    for l in range(4):
     s=[forcing[idx(l,k+2,i+j)]if axis==0 else forcing[idx(l,k+j,i+2)]for j in range(4)]
     vals.append(math.fsum(a*b for a,b in zip([-1/12,7/12,7/12,-1/12],s)));third.append(math.fsum(a*b for a,b in zip([-1,3,-3,1],s)))
    r=vals[0]+(rho[k+2]if axis==0 else ri[k]);u=vals[1]/r;w=vals[2]/r;th=(vals[3]+(rt[k+2]if axis==0 else rti[k]))/r;pr=pressure(r*th)
    if axis==0:physical=[r*u,r*u*u+pr,r*u*w,r*u*th]
    else:
     pr-=pi[k]
     if k==0 or k==nz:w=0.;third[0]=0.
     physical=[r*w,r*w*u,r*w*w+pr,r*w*th]
    for l in range(4):flux[fx(l,k,i)]=physical[l]-hv*third[l]
  for l in range(4):
   for k in range(nz):
    for i in range(nx):
     v=-(flux[fx(l,k,i+1)]-flux[fx(l,k,i)])/int(dx)if axis==0 else -(flux[fx(l,k+1,i)]-flux[fx(l,k,i)])/int(dz)
     if axis==1 and l==2:v-=forcing[idx(0,k+2,i+2)]*g
     tend[tx(l,k,i)]=v
  for l in range(4):
   for k in range(nz):
    for i in range(nx):out[idx(l,k+2,i+2)]=state[idx(l,k+2,i+2)]+subdt*tend[tx(l,k,i)]
 for step,dt in enumerate(c['schedule']):
  for axis in ([0,1]if step%2==0 else[1,0]):
   stage(state,tmp,dt/3,axis,dt);stage(tmp,tmp,dt/2,axis,dt);stage(tmp,state,dt,axis,dt)
 diag=initial_diagnostics+diagnostics(state)
 mass0,_,mass1,_=diag
 assert abs(mass1-mass0)<=1e-12*mass0
 if c['kind']=='rest':assert max(abs(v)for v in state)<2e-8
 return [state,tmp,flux,tend,sl,sr,rl,rr,diag]
cases=[make(16,8,1,'rest'),make(32,16,2,'thermal'),make(40,20,3,'moving_thermal')];f=[len(cases)];o=[];info=[]
for c in cases:
 out=solve(c);f +=[c['nx'],c['nz'],c['steps'],c['dx'],c['dz'],c['dt']]+c['schedule']+c['initial']+[v for a in c['hy']for v in a];o +=[v for a in out for v in a];info.append({'nx':c['nx'],'nz':c['nz'],'steps':c['steps'],'kind':c['kind'],'schedule':c['schedule'],'output_lengths':[len(a)for a in out],'initial_final_mass':out[-1][::2],'max_final_abs_state':max(map(abs,out[0]))})
(p/'fixtures_f64.bin').write_bytes(struct.pack('<'+'d'*len(f),*f));(p/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(o),*o));(p/'independent_oracle.json').write_text(json.dumps({'method':'IndependentPythonfinitevolumeEulerphysicalfluxes, matrixcoefficientinterpolation/math.fsum,Decimal50EOSpressure,fullsplit3stageRK+periodicandverticalsourceboundarypolicy; no nativeheaders. Sourcefullyglobaldtforhyperviscosity preserved, no stagedtcoefficient. All8mutablearrays+4diagnosticvalues oracle.','cases':info,'input_words':len(f),'oracle_words':len(o),'source_energy_diagnostic':'Sourcekineticterm rho*(u^2+w^2) no1/2, nottextbooktotalenergy/conservationcertificate.','tolerance':{'atol':2e-8,'rtol':2e-11},'scope':'Finite1/2/3steps/syntheticrest/thermal/movingthermal, exactintegerdx/dz(1250,625,500), scalarFloat64oraclesamplesfullstate; notconvergence/weatherprediction/600secCLI'},indent=2)+'\n');print({'input_words':len(f),'oracle_words':len(o),'cases':info})
