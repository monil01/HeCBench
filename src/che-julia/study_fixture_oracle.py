from pathlib import Path
import math, struct, json
p=Path(__file__).resolve().parent
cases=[(16,8,8,1,0),(19,9,7,3,0),(16,8,8,3,1),(16,8,8,3,2)];fixture=[];oracle=[];mass=[];energies=[]
for nx,ny,nz,steps,mode in cases:
 v=nx*ny*nz;c=[]
 for z in range(nz):
  for y in range(ny):
   for x in range(nx):
    c.append(((x*17+y*11+z*7)%31-15)/64 if mode==0 else .25 if mode==1 else math.cos(2*math.pi*x/nx)/16)
 fixture.extend(c);initialmass=math.fsum(c)
 def at(x,y,z):return (z%nz*ny+y%ny)*nx+x%nx
 def lap(a,x,y,z):return math.fsum(a[at(x+dx,y+dy,z+dz)]-a[at(x,y,z)] for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
 for st in range(steps):
  mu=[];ff=[]
  for z in range(nz):
   for y in range(ny):
    for x in range(nx):
     i=at(x,y,z);q=c[i];mu.append(q*q*q-q-.5*lap(c,x,y,z));grads=[(c[at(x+dx,y+dy,z+dz)]-c[at(x-dx,y-dy,z-dz)])/2 for dx,dy,dz in [(1,0,0),(0,1,0),(0,0,1)]];ff.append(.25*q**4-.5*q*q+.25*sum(g*g for g in grads))
  cn=[c[at(x,y,z)]+.01*lap(mu,x,y,z) for z in range(nz)for y in range(ny)for x in range(nx)];old=c;c=cn
 oracle.extend(c+old+mu+ff);mass.append({'initial':initialmass,'final':math.fsum(c),'absolute_drift':abs(math.fsum(c)-initialmass)});energies.append(math.fsum(ff))
for name,data in [('fixtures_f64.bin',fixture),('oracle_f64.bin',oracle)]:
 with(p/name).open('wb')as io:io.write(struct.pack('<'+'d'*len(data),*data))
(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent simplified free-energy derivative mu=c^3-c-(1/2)Laplacian(c), f=(1/4)c^4-(1/2)c^2+(1/4)|centered-gradient(c)|²; periodic neighbor-difference fsum, explicit Euler. Not a literal translation of source coefficient expression.','source_coefficient_identity':'eAA=eBB=-2/9,eAB=2/9 reduce local potential to (1/4)c^4-(1/2)c² mathematically; actual rounded source coefficients compared at declared1e-11 bound','cases':cases,'mass_conservation':mass,'max_mass_drift':max(x['absolute_drift']for x in mass),'constant_state_exact_expected':{'c':.25,'mu':-.234375,'f':-.0302734375},'scientific_scope':'Discrete periodic conserved-phase operator and short-step full fields, not256cubed coarsening or physical-material validation'},indent=2)+'\n')
