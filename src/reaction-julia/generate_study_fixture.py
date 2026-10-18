from pathlib import Path
import json,struct
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
pre=[4];meta=[]
for ci,(M,steps,kind) in enumerate([(16,5,'nonuniform_gray_scott'),(32,4,'central_cube_gray_scott'),(16,3,'exact_equilibrium'),(16,1,'manufactured_diffusion_only')],1):
 N=M**3;a=[];b=[];coords=[(i%M,(i//M)%M,i//(M*M)) for i in range(N)]
 for x,y,z in coords:
  if ci==1:aa=.75+((x+3*y+5*z)%17)/128;bb=.125+((7*x+5*y+z)%9)/256
  elif ci==2:
   core=all(M//2-5<=v<M//2+5 for v in [x,y,z]);aa=.5 if core else 1.;bb=.25 if core else 0.
  elif ci==3:aa=1.;bb=0.
  else:aa=.5+x/64+y/128+z/256;bb=0.
  a.append(f32(aa));b.append(f32(bb))
 ca=f32(f32(.16)/f32(.5*.5));cb=f32(f32(.08)/f32(.5*.5));dt=f32(.25);c1=f32(0 if ci==4 else .0392);c2=f32(0 if ci==4 else .0649);params=[ca,cb,dt,c1,c2]
 pre.extend([M,steps]+params);pre.extend(a);pre.extend(b)
 neighbors=[[(i-1 if x>0 else i,i+1 if x<M-1 else i),(i-M if y>0 else i,i+M if y<M-1 else i),(i-M*M if z>0 else i,i+M*M if z<M-1 else i)] for i,(x,y,z) in enumerate(coords)]
 def derivatives(f):return [[f[neighbors[i][axis][0]]+f[neighbors[i][axis][1]]-2*f[i] for i in range(N)] for axis in range(3)]
 trajectory=[]
 for step in range(1,steps+1):
  ad=derivatives(a);bd=derivatives(b);da=[ca*(ad[0][i]+ad[1][i]+ad[2][i]) for i in range(N)];db=[cb*(bd[0][i]+bd[1][i]+bd[2][i]) for i in range(N)];r=[a[i]*b[i]*b[i] for i in range(N)];ra=[-r[i]+c1*(1-a[i]) for i in range(N)];rb=[r[i]-(c1+c2)*b[i] for i in range(N)];before=sum(a)+sum(b);a=[a[i]+dt*(da[i]+ra[i]) for i in range(N)];b=[b[i]+dt*(db[i]+rb[i]) for i in range(N)]
  trajectory.append(dict(step=step,mean_A=sum(a)/N,mean_B=sum(b)/N,mean_diffusion_A=sum(da)/N,mean_diffusion_B=sum(db)/N,combined_mass_balance_residual=(sum(a)+sum(b)-before-dt*(sum(ra)+sum(rb)))/N,min_A=min(a),max_A=max(a),min_B=min(b),max_B=max(b)))
 fields=[a,b,*bd,ra,rb,da,db];gold=[v for f in fields for v in f];(p/f'oracle_case{ci}.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));meta.append(dict(case=ci,M=M,N=N,steps=steps,kind=kind,parameters=params,oracle_fields=['A','B','B_dxx','B_dyy','B_dzz','reaction_A','reaction_B','diffusion_A','diffusion_B'],cpu_float64_trajectory=trajectory,oracle_values=len(gold)))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'fixture_description.json').write_text(json.dumps(dict(cases=meta,input_prefix_values=len(pre),boundary='Zero-flux repeated center at outer faces; full source axis-pencil shared tiles. Selected cubic M16/32 and pencils2.',oracle='Independent Float64 global-neighbor finite-difference GrayScott Euler with actual Float32 input parameters; source Float32 c1+c2/coefficient arithmetic rounding accepted by fixed budget. No shared tiling/cache or GPU code in oracle.',acceptance=dict(atol=1e-6,rtol=5e-6,exact_equilibrium=True,exact_actual_input_bits_and_guards=True,nonnegative_species=True,diffusion_mean_absolute_budget=1e-6),source_scope='Six selected zero-flux/reaction/laplacian/update kernels,10 launches per timestep/13 total steps. Not periodic branch, default128cubed10000step morphology or continuum convergence.'),indent=2)+'\n');print(len(pre),sum(c['N'] for c in meta),sum(c['steps'] for c in meta))
