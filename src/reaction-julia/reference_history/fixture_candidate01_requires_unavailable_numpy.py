from pathlib import Path
import numpy as np,json,struct
p=Path(__file__).resolve().parent;pre=[4];meta=[]
for ci,(M,steps,kind) in enumerate([(16,5,'nonuniform_gray_scott'),(32,4,'central_cube_gray_scott'),(16,3,'exact_equilibrium'),(16,1,'manufactured_diffusion_only')],1):
 z,y,x=np.indices((M,M,M));N=M**3
 if ci==1:a=(.75+((x+3*y+5*z)%17)/128).astype(np.float32);b=(.125+((7*x+5*y+z)%9)/256).astype(np.float32)
 elif ci==2:
  core=(abs(x-M//2)<3)&(abs(y-M//2)<3)&(abs(z-M//2)<3);a=np.where(core,.5,1).astype(np.float32);b=np.where(core,.25,0).astype(np.float32)
 elif ci==3:a=np.ones((M,M,M),np.float32);b=np.zeros_like(a)
 else:a=(.5+x/64+y/128+z/256).astype(np.float32);b=np.zeros_like(a)
 ca=np.float32(.16)/np.float32(.5*.5);cb=np.float32(.08)/np.float32(.5*.5);dt=np.float32(.25);c1=np.float32(0 if ci==4 else .0392);c2=np.float32(0 if ci==4 else .0649)
 params=[float(v) for v in [ca,cb,dt,c1,c2]];pre.extend([M,steps]+params);pre.extend(a.ravel().astype(float));pre.extend(b.ravel().astype(float));a=a.astype(float);b=b.astype(float);trajectory=[]
 def derivatives(f):
  result=[]
  for axis in [2,1,0]:
   hi=np.take(f,np.minimum(np.arange(M)+1,M-1),axis=axis);lo=np.take(f,np.maximum(np.arange(M)-1,0),axis=axis);result.append(hi+lo-2*f)
  return result
 for step in range(1,steps+1):
  ad=derivatives(a);bd=derivatives(b);da=float(ca)*sum(ad);db=float(cb)*sum(bd);r=a*b*b;ra=-r+float(c1)*(1-a);rb=r-(float(c1)+float(c2))*b
  before=float(np.sum(a+b));a=a+float(dt)*(da+ra);b=b+float(dt)*(db+rb)
  trajectory.append(dict(step=step,mean_A=float(np.mean(a)),mean_B=float(np.mean(b)),mean_diffusion_A=float(np.mean(da)),mean_diffusion_B=float(np.mean(db)),combined_mass_balance_residual=(float(np.sum(a+b))-before-float(dt)*float(np.sum(ra+rb)))/N,min_A=float(a.min()),max_A=float(a.max()),min_B=float(b.min()),max_B=float(b.max())))
 fields=[a,b,*bd,ra,rb,da,db];gold=np.concatenate([f.ravel() for f in fields]);(p/f'oracle_case{ci}.bin').write_bytes(gold.astype('<f8').tobytes());meta.append(dict(case=ci,M=M,N=N,steps=steps,kind=kind,parameters=params,oracle_fields=['A','B','B_dxx','B_dyy','B_dzz','reaction_A','reaction_B','diffusion_A','diffusion_B'],cpu_float64_trajectory=trajectory,oracle_values=len(gold)))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'fixture_description.json').write_text(json.dumps(dict(cases=meta,input_prefix_values=len(pre),boundary='Zero-fluxrepeatedcenteratouterfaces; fullsourceaxis-pencilsharedtiles. SelectedcubicM16/32andpencils2.',oracle='IndependentFloat64globalneighborfinite-differenceGrayScottEuler withactualFloat32inputparameters; sourceFloat32c1+c2/coefficientarithmeticroundingacceptedbyfixedbudget. No sharedtiling/cacheorGPUcode inoracle.',acceptance=dict(atol=1e-6,rtol=5e-6,exact_equilibrium=True,exact_actual_input_bits_and_guards=True,nonnegative_species=True,diffusion_mean_absolute_budget=1e-6),source_scope='Sixselectedzero-flux/reaction/laplacian/updatekernels,10launchespertimestep/13totalsteps. Notperiodicbranch,default128cubed10000stepmorphologyorcontinuumconvergence.'),indent=2)+'\n');print(len(pre),sum(c['N'] for c in meta),sum(c['steps'] for c in meta))
