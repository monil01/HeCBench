from pathlib import Path
import struct,sys,math,json,hashlib
p=Path(__file__).resolve().parent
fixture=(p/'fixtures_f32.bin').read_bytes();pos=4;cases=[]
for ci in range(3):
 nx,ny,dt=struct.unpack_from('<IIf',fixture,pos);pos+=12;a=[]
 for f in range(15):
  n=nx*ny*(3 if f<10 else 1);a.append(list(struct.unpack_from('<%df'%n,fixture,pos)));pos+=4*n
 cases.append((nx,ny,dt,a))
raw=Path(sys.argv[1]).read_bytes();v=struct.unpack('<%dd'%(len(raw)//8),raw);assert len(v)==212480
idx=0;assert v[idx]==3;idx+=1
for nx,ny,dt,a in cases:
 assert list(v[idx:idx+3])==[nx,ny,dt];idx+=3
 for x in a:assert list(v[idx:idx+len(x)])==x;idx+=len(x)
assert idx==103365
oracle=(p/'oracle_cpu_f32.bin').read_bytes();o=struct.unpack('<%df'%(len(oracle)//4),oracle);oi=0;ma=mr=ms=0.;reviews=[]
for ci,(nx,ny,dt,a)in enumerate(cases):
 b=[]
 for f in range(15):
  n=len(a[f]);y=list(v[idx:idx+n]);idx+=n;guards=v[idx:idx+128];idx+=128;assert all(x==-777 for x in guards);assert all(map(math.isfinite,y))
  if f in [1,7,8,9]:assert y==a[f]
  for x,z in zip(y,o[oi:oi+n]):
   e=abs(x-z);ma=max(ma,e);mr=max(mr,e/max(abs(z),1e-12));ms=max(ms,e/(1e-7+5e-5*abs(z)))
  oi+=n;b.append(y)
 assert all(x>=0 for f in range(2,7)for x in b[f]);assert all(-1e-7<=x<=1+1e-7 for x in b[12])
 if ci==0:
  for f in range(15):assert b[f]==([0.]*len(b[f]) if f in [11,12,14] else a[f])
 residual=scaled=0.;u=2**-24;gamma=4096*u/(1-4096*u)
 for j in range(ny):
  for i in range(nx):
   inds=[i+nx*k+3*nx*j for k in range(3)];col=i+nx*j
   w0=sum(a[7][k]*a[9][k]*sum(a[f][k]for f in range(2,7))for k in inds);w1=sum(b[7][k]*b[9][k]*sum(b[f][k]for f in range(2,7))for k in inds)
   err=abs(w0-w1-(b[10][col]-a[10][col]));budget=gamma*(w0+abs(a[10][col])+1)+3*max(a[7][k]for k in inds)*500*1e-12
   residual=max(residual,err);scaled=max(scaled,err/budget)
 assert scaled<=1
 reviews.append({'case':ci,'max_water_residual_kg_m2':residual,'max_water_normalized_budget':scaled,'new_ice_positive_cells':sum(x>0 for x in b[4]),'guards_readonly_dry_checks':True})
assert idx==len(v) and oi==len(o) and ms<=1
review={'artifact_sha256':hashlib.sha256(raw).hexdigest(),'full_source_CPU_oracle_max_abs':ma,'max_rel_floor_1e12':mr,'max_normalized_budget':ms,'cases':reviews,'scope':'Conditional two-flux-init source, initialqi0/uniformdz/3levels/mstep1; no global climate/rate certification'}
Path(sys.argv[2]).write_text(json.dumps(review,indent=2)+'\n');Path(sys.argv[2]+'.metrics').write_bytes(struct.pack('<ddd',ma,mr,ms))
