from pathlib import Path
import argparse,json,hashlib
import numpy as np
R=Path('/home/imo/HeCBench_julia');B=R/'src/lulesh-julia';layout=json.loads((B/'study_layout.json').read_text());fields=layout['fields'];cases=layout['cases'];params=layout['parameters'];G=64
p=argparse.ArgumentParser();p.add_argument('--artifact');p.add_argument('--result');a=p.parse_args();rows=[]
if a.result:
 result=json.loads(Path(a.result).read_text());items=[(m,t['trial'],Path(t['output_artifact']['saved_path']))for m,v in result['models'].items()for t in v['trials']]
else:items=[('serial_source_CPU',0,Path(a.artifact))]
readonly={'nodelist','volo','nodeElemStart','nodeElemCornerList','gamma','elemMass','nodalMass','symmX','symmY','symmZ','lzetam','lzetap','letam','letap','lxim','lxip','elemBC','elemRep','elemElem'}
signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float);faces=np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
def hashfile(path):
 with path.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(actual,want,label,atol=1e-7,rtol=3e-8):
 error=np.abs(actual-want);budget=atol+rtol*np.abs(want);assert np.all(np.isfinite(actual))and np.all(error<=budget),(label,float(np.max(error)),float(np.max(error/np.maximum(budget,1e-300))));return float(np.max(error,initial=0))
def volume(x):
 out=np.zeros(len(x))
 for pt in signs/np.sqrt(3.):
  grad=np.empty((8,3))
  for dim in range(3):
   others=[k for k in range(3)if k!=dim];grad[:,dim]=signs[:,dim]*(1+signs[:,others[0]]*pt[others[0]])*(1+signs[:,others[1]]*pt[others[1]])/8
  jac=np.einsum('eni,nj->eij',x,grad);out+=np.linalg.det(jac)
 return out
for model,trial,path in items:
 v=np.memmap(path,dtype='<f8',mode='r');assert len(v)==layout['artifact_words'];off=0;pre=[]
 def state(i):
  global off
  d={}
  for f in fields:
   n=f['case_lengths'][i];z=v[off:off+n+2*G];off+=n+2*G;guard=-987654321.125 if f['type']=='Float64'else-987654321
   assert np.all(z[:G]==guard)and np.all(z[-G:]==guard),(model,trial,f['name'],'typedrawguard');d[f['name']]=z[G:-G]
  return d
 for i,c in enumerate(cases):
  st=state(i);raw=(B/f'case{i}.bin').read_bytes();bpos=0
  for f in fields:
   dt='<f8'if f['type']=='Float64'else'<i4';want=np.frombuffer(raw,dtype=dt,count=f['case_lengths'][i],offset=bpos);bpos+=want.nbytes;assert np.array_equal(st[f['name']],want),(model,trial,'actualinput',f['name'])
  ps=v[off:off+len(params)];off+=len(params);assert np.array_equal(ps,np.fromfile(B/f'params{i}.bin',dtype='<f8'))
  reg=np.fromfile(B/f'regions{i}.bin',dtype='<i4');assert np.array_equal(v[off:off+len(reg)],reg);off+=len(reg);pre.append((st,ps))
 assert off==layout['input_prefix_words'];details=[]
 for i,c in enumerate(cases):
  prev,ps=pre[i];par=dict(zip(params,ps));idx=prev['nodelist'].astype(np.int64).reshape(-1,8);prev_time=par['time'];previous_dt=par['deltatime']
  for step in range(c['steps']):
   d=state(i);time,dt,cycle,dc,dh=v[off:off+5];off+=5;assert cycle==step+1;assert d['vol_error'][0]==-1
   for n in readonly:assert np.array_equal(d[n],pre[i][0][n]),(model,trial,n,'readonly')
   X=np.stack([d[n]for n in ['x','y','z']],axis=-1);U=np.stack([d[n]for n in ['xd','yd','zd']],axis=-1);xe=X[idx];ue=U[idx];vol=volume(xe);assert np.all(vol>0)
   verr=close(vol,d['volo']*d['vnew'],'Gauss_integrated_hexahedron_volume');deterr=close(d['determ'],prev['volo']*prev['v'],'actual_old_volume_hgc_determ')
   mid=xe-.5*dt*ue;J=np.einsum('eni,nj->eij',mid,signs/8);VJ=np.einsum('eni,nj->eij',ue,signs/8);grad=VJ@np.linalg.inv(J);trace=np.trace(grad,axis1=1,axis2=2);strainerr=close(d['vdov'],trace,'independent_isoparametric_divergence')
   close(d['dxx']+d['dyy']+d['dzz'],np.zeros(len(vol)),'deviatoric_trace')
   fv=xe[:,faces,:];diag1=fv[:,:,2,:]-fv[:,:,0,:];diag2=fv[:,:,3,:]-fv[:,:,1,:];maxarea=np.max(np.linalg.norm(np.cross(diag1,diag2),axis=-1),axis=1);char=2*vol/maxarea;cerr=close(d['arealg'],char,'face_cross_product_characteristic_length')
   wantp=(2/3)*d['e']/d['vnew'];wantp=np.where(np.abs(wantp)<par['p_cut'],0,wantp);wantp=np.maximum(wantp,par['pmin']);perr=close(d['p'],wantp,'ideal_gas_pressure')
   ss2=((2/3)*d['e']+d['vnew']**2*((2/3)/d['vnew'])*d['p'])/par['refdens'];wantss=np.sqrt(np.maximum(ss2,0));wantss=np.where(ss2<=.1111111e-36,.3333333e-18,wantss);serr=close(d['ss'],wantss,'sound_speed_closure')
   wantq=np.where(d['delv']>0,0,d['ss']*d['ql']+d['qq']);wantq=np.where(np.abs(wantq)<par['q_cut'],0,wantq);qerr=close(d['q'],wantq,'final_artificial_viscosity');assert np.all(d['ql']>=-1e-12)and np.all(d['qq']>=-1e-12)
   for name,vel,acc,force,sym in [('x','xd','xdd','fx','symmX'),('y','yd','ydd','fy','symmY'),('z','zd','zdd','fz','symmZ')]:
    want=d[force]/d['nodalMass'];want[d[sym].astype(np.int64)]=0;close(d[acc],want,'nodal_acceleration_'+acc)
    wvel=prev[vel]+d[acc]*dt;wvel=np.where(np.abs(wvel)<par['u_cut'],0,wvel);close(d[vel],wvel,'velocity_'+vel);close(d[name],prev[name]+d[vel]*dt,'position_'+name)
   mask=d['vdov']!=0;dtf=d['ss']**2+np.where(d['vdov']<0,64*par['qqc']**2*d['arealg']**2*d['vdov']**2,0);wc=min(1e20,float(np.min(d['arealg'][mask]/np.sqrt(dtf[mask]),initial=1e20)));wh=min(1e20,float(np.min(par['dvovmax']/(np.abs(d['vdov'][mask])+1e-20),initial=1e20)));close(np.array([dc,dh]),np.array([wc,wh]),'global_Courant_hydro_min')
   close(np.array([time]),np.array([prev_time+dt]),'time_increment',1e-14,1e-13);assert time<=par['stoptime']+1e-14 and dt>0
   if c['mode']:
    alpha=0 if c['mode']==1 else .01 if c['mode']==2 else-.01
    for n in ['x','y','z']:close(d[n],pre[i][0][n]*(1+alpha*dt),'manufactured_affine_'+n,1e-12,1e-12)
    close(d['vnew'],np.full(len(vol),(1+alpha*dt)**3),'manufactured_affine_relative_volume',1e-12,1e-12)
   details.append(dict(case=i,label=c['label'],step=step+1,time=float(time),dt=float(dt),Gauss_volume_max_abs=verr,old_volume_max_abs=deterr,divergence_max_abs=strainerr,characteristic_length_max_abs=cerr,pressure_closure_max_abs=perr,sound_closure_max_abs=serr,viscosity_closure_max_abs=qerr,minimum_volume=float(vol.min()),specific_internal_energy_min=float(d['e'].min()),specific_internal_energy_max=float(d['e'].max()),sum_internal_energy_mass=float(np.dot(d['e'],d['elemMass'])),scope='State/checkpoint and physical algebra invariants, not full Sedov analytic convergence/total-energy conservation certification'))
   prev=d;prev_time=time;previous_dt=dt
 assert off==len(v);rows.append(dict(model=model,trial=trial,path=str(path),sha256=hashfile(path),cases=details))
report=dict(all_pass=True,artifacts=len(rows),artifact_words=layout['artifact_words'],input_prefix_words=layout['input_prefix_words'],checks='Every actual typed initial input/metadata/guard and full final cycle state. Independent Gauss determinant volume, isoparametric gradient, face cross product, nodal update, ideal-gas pressure/sound/viscosity, global timeconstraint minima, analytic rest/affine controls. Source serial full-state comparator is separate common-source corroboration, not an independent scientific oracle.',numpy=np.__version__,trials=rows)
out=B/('output_audit.json'if a.result else'CPU_physical_invariant_audit.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('Independent geometric/physical invariants PASS',len(rows),'complete artifacts')
