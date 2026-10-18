"""Source-based cavity fixture and independent vector-form D3Q19 CPU oracle; no GPU."""
from pathlib import Path
import struct,math,json,hashlib,time
p=Path(__file__).resolve().parent
N=17;NL=N**3;STEPS=20;LID=struct.unpack('<f',struct.pack('<f',.02))[0];OMEGA=1/(3*(.02*(N-2)/100)+.5)
C=[(0,0,0),(-1,0,0),(0,-1,0),(0,0,-1),(-1,-1,0),(-1,1,0),(-1,0,-1),(-1,0,1),(0,-1,-1),(0,-1,1),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,-1,0),(1,0,1),(1,0,-1),(0,1,1),(0,1,-1)]
W=[12/36]+[2/36]*3+[1/36]*6+[2/36]*3+[1/36]*6
F32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
idx=lambda x,y,z:(x%N)+((y%N)+(z%N)*N)*N
base=[0]*NL;dirs=[0]*NL;where=[0]*NL
for z in range(N):
 for y in range(N):
  for x in range(N):
   i=idx(x,y,z)
   if x in [0,N-1] or y in [0,N-1] or z in [0,N-1]:base[i]=4
   if x in [0,N-1] or y in [0,N-1] or z==N-1:where[i]=4
   if z==0:base[i]=10;dirs[i]=4
flags=base.copy()
for z in range(N):
 for y in range(N):
  for x in range(N):
   i=idx(x,y,z)
   if base[i]>=4:continue
   for q,(cx,cy,cz) in enumerate(C):
    if not q:continue
    neighbor=idx(x+cx,y+cy,z+cz)
    if base[neighbor] in [4,10]:
     opposite=(q+8)%18+1;where[i]|=1<<opposite
     if base[neighbor]==10:flags[i]=2;dirs[i]=dirs[neighbor]
     elif flags[i]!=2:flags[i]=3
initial=[W[q] for q in range(19) for i in range(NL)]
prefix=initial+list(map(float,flags))+list(map(float,dirs))+list(map(float,where))+[float(N),float(STEPS),float(LID),OMEGA]
raw=struct.pack('<'+'d'*len(prefix),*prefix);(p/'fixture.bin').write_bytes(raw)
f0=initial.copy();f1=initial.copy();vel=[[0.]*NL for _ in range(3)];active=[i for i,f in enumerate(flags) if f<4];history=[]
for step in range(STEPS):
 for i in active:
  x=i%N;y=(i//N)%N;z=i//(N*N);fin=[]
  for q,(cx,cy,cz) in enumerate(C):
   val=f0[q*NL+idx(x-cx,y-cy,z-cz)]
   if where[i]&(1<<q):val=f0[((q+8)%18+1)*NL+i]
   if flags[i]==2 and cz==1:val+=6*W[q]*(cx*LID)
   fin.append(val)
  rho=math.fsum(fin);u=[F32(math.fsum(fin[q]*C[q][a] for q in range(19))/rho) for a in range(3)];u2=math.fsum(v*v for v in u)
  for q,(cx,cy,cz) in enumerate(C):
   dot=cx*u[0]+cy*u[1]+cz*u[2];eq=rho*W[q]*(1+3*dot+4.5*dot*dot-1.5*u2);f1[q*NL+i]=(1-OMEGA)*fin[q]+OMEGA*eq
  for a in range(3):vel[a][i]=u[a]
 f0,f1=f1,f0
 mass=math.fsum(f0[q*NL+i] for q in range(19) for i in active);history.append({'step':step+1,'active_mass':mass,'mass_relative_drift':abs(mass-len(active))/len(active),'min_population':min(f0),'max_velocity':max(abs(x) for v in vel for x in v)})
# Source data layouts/actual native guards, including immutable rho and boundary metadata.
gold=[]
for arr in [f0,f1,[v for component in vel for v in component],[1.]*NL]:gold.extend([-12345.]*8+arr+[-12345.]*8)
for arr in [flags,dirs,where]:gold.extend([-991827.]*8+list(map(float,arr))+[-991827.]*8)
(p/'oracle.bin').write_bytes(struct.pack('<'+'d'*len(gold),*gold))
meta={'configuration':'moving_wall_cavity17_20steps','N':N,'nl':NL,'steps':STEPS,'lid_Float32_widened':LID,'omega':OMEGA,'prefix_count':len(prefix),'output_count':len(gold),'artifact_count':len(prefix)+len(gold),'input_sha256':hashlib.sha256(raw).hexdigest(),'oracle_sha256':hashlib.sha256((p/'oracle.bin').read_bytes()).hexdigest(),'source_initialization_correction':'Initialize boundary_values and boundary_dirs to0 BEFORE unchanged make_flag/find_wall; source allocation doesnotinitialize these and find_wall reads|= onvalues. Static source diagnosis, not observed undefined GPU run.','model':'source17cubecavity, allwallsbounce except moving zmin; no2000stepCLIclaim','independent_oracle':'GenericD3Q19vector pull/bounce/moving-wall +math.fsum moments andFloat32velocities +standardsecondordervectorequilibrium. GPU sourceoptimizedmixed-F32subexpressions may differ; predeclared atol1e-7 rtol1e-6, not formal trajectory errorbound.','guard_policy':'f0/f1 outer16 each, contiguous3componentu_star outer16, rho outer16; metadata flags/dirs/where outer16 each. Charflag/dir guard physical8bit mustchoose representable−83, Int32where−991827; oraclechar guards correctedbelow.','source_flags_counts':{str(v):flags.count(v) for v in set(flags)},'active_cells':len(active),'history':history,'precision':'Float64populations; Float32macroscopicvelocities; sourceequilibriumFloat32velocitysubexpressions thenFloat64; independentCPU mathematicalformulation inFloat64 afterF32velocity conversion','initialization_only_native_reference_repaired':True,'periodic_control_status':'separate secondary planned, not pooled'}
# Metadata char buffer guards are representable signed-char -83, separate from Int32 where guards.
off=(2*19*NL+3*NL+NL)+64
for a in range(2):
 start=off+a*(NL+16);gold[start:start+8]=[-83.]*8;gold[start+8+NL:start+16+NL]=[-83.]*8
(p/'oracle.bin').write_bytes(struct.pack('<'+'d'*len(gold),*gold));meta['oracle_sha256']=hashlib.sha256((p/'oracle.bin').read_bytes()).hexdigest()
(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:meta[k] for k in ['prefix_count','output_count','artifact_count','source_flags_counts','active_cells']}));print(history[-1])
