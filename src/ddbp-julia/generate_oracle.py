"""Global-prefix and ray/plane geometry oracle for scoped corrected backprojection."""
from pathlib import Path
import numpy as np,json,hashlib,sys,math
P=Path(__file__).resolve().parent
cases=[dict(name='analytic_unit_plane',nx=8,ny=8,ns=1,ndx=8,ndy=8,np=1,dx=1.,dy=1.,dz=1.,du=1.,dv=1.,DSD=8.,DDR=0.,DAG=-.5,tube=[0.],det=[0.]),dict(name='varied_three_views_two_slices',nx=7,ny=8,ns=2,ndx=8,ndy=8,np=3,dx=.9,dy=1.,dz=.5,du=1.,dv=1.,DSD=20.,DDR=2.,DAG=1.,tube=[-5.,0.,4.],det=[-2.,.5,1.5])]
allmeta=[]
for ci,p in enumerate(cases,1):
 X,Y=p['ndx']+1,p['ndy']+1;OX,OY=p['nx']+1,p['ny']+1
 shape=(p['np'],p['ndx'],p['ndy']);idx=np.indices(shape);proj=np.ones(shape) if ci==1 else (16+3*idx[0]+2*idx[1]+idx[2]+idx[1]*idx[2]%7)/16
 padded=np.zeros((p['np'],X,Y));padded[:,1:,1:]=proj
 detx=(p['ndx']-np.arange(X))*p['du'];dety=(np.arange(Y)-p['ndy']/2)*p['dv'];detz=np.zeros(Y);objx=(p['nx']-np.arange(OX))*p['dx'];objy=(np.arange(OY)-p['ny']/2)*p['dy'];objz=p['DAG']+(np.arange(p['ns'])+.5)*p['dz'];prefix=padded.ravel().tolist()+p['tube']+p['det']+[p[k] for k in ['nx','ny','ns','ndx','ndy','np','dx','dy','dz','du','dv','DSD','DDR','DAG']]
 states=[];labels=[]
 def capture(label,*arrays):
  labels.append(dict(label=label,start=sum(len(z) for z in states),count=sum(np.asarray(a).size for a in arrays)));states.append(np.concatenate([np.asarray(a).ravel() for a in arrays]))
 capture('all_detector_object_boundaries',detx,dety,detz,objx,objy,objz)
 integralx=np.cumsum(padded,axis=1);capture('integral_after_X',integralx)
 integral=np.cumsum(integralx,axis=2);capture('integral_after_Y',integral)
 volume=np.zeros((p['ns'],p['nx'],p['ny']))
 for view,(ta,da) in enumerate(zip(p['tube'],p['det'])):
  theta=math.radians(ta);phi=math.radians(da)
  source_y=-(p['DSD']-p['DDR'])*math.sin(theta);source_z=(p['DSD']-p['DDR'])*math.cos(theta)+p['DDR']
  ry=dety*math.cos(phi)+p['DDR']*math.sin(phi);rz=dety*math.sin(phi)-p['DDR']*math.cos(phi)+p['DDR'];capture(f'view{view}_rotated_detector',ry,rz)
  for sl,z in enumerate(objz):
   # Ray from source(0,source_y,source_z) through detector point;
   # t=(z-source_z)/(detector_z-source_z), independent vector line-plane form.
   t=(z-source_z)/(rz-source_z);mappedx=detx[:,None]*t[None,:];mappedy=source_y+t*(ry-source_y);capture(f'view{view}_slice{sl}_mapped',mappedx,mappedy)
   step=mappedx[p['ndx']-1,0];qx=p['ndx']-objx/step;qy=objy/step-mappedy[0]/step;sample=np.zeros((OX,OY))
   for ix,x in enumerate(qx):
    for iy,y in enumerate(qy):
     bx,by=math.floor(x),math.floor(y);a,b=x-bx,y-by
     def at(xi,yi,valid):return integral[view,xi,yi] if valid else 0.
     v00=at(bx,by,0<=x<=p['ndx'] and 0<=y<=p['ndy'])
     v10=at(bx+1,by,0<bx+1<=p['ndx'] and 0<=y<=p['ndy'])
     v01=at(bx,by+1,0<=x<=p['ndx'] and 0<by+1<=p['ndy'])
     v11=at(bx+1,by+1,0<bx+1<=p['ndx'] and 0<by+1<=p['ndy'])
     sample[ix,iy]=(1-a)*(1-b)*v00+a*(1-b)*v10+(1-a)*b*v01+a*b*v11
   capture(f'view{view}_slice{sl}_interpolated_integral',sample)
   A=sample[:-1,:-1];B=sample[1:,:-1];C=sample[:-1,1:].copy();D=sample[1:,1:].copy();mask=(C==0)&(D==0);C[mask]=A[mask];D[mask]=B[mask]
   gx=(objx[:-1]+p['dx']/2)/ (source_z-z);gy=(objy[:-1]+p['dy']/2-source_y)/(source_z-z)
   factor=p['du']*p['dv']*p['dz']/(p['dx']*p['dy'])*np.sqrt(1+gx[:,None]**2)*np.sqrt(1+gy[None,:]**2)
   volume[sl]+=(D-C-B+A)*factor;capture(f'view{view}_slice{sl}_cumulative_volume',volume)
 volume/=p['np'];capture('normalized_volume',volume)
 if ci==1:
  analytic=np.array([[math.sqrt(1+((8-x+.5)/8)**2)*math.sqrt(1+((y-4+.5)/8)**2) for y in range(8)] for x in range(8)])
  assert np.max(np.abs(volume[0]-analytic))<1e-14
 oracle=np.concatenate(states);input_bytes=proj.astype('<f8').tobytes();(P/f'study_proj{ci}_f64.bin').write_bytes(input_bytes);raw=oracle.astype('<f8').tobytes();(P/f'oracle_case{ci}_f64.bin').write_bytes(raw)
 allmeta.append(dict(case=ci,geometry=p,prefix=prefix,state_labels=labels,oracle_count=len(oracle),input_sha256=hashlib.sha256(input_bytes).hexdigest(),oracle_sha256=hashlib.sha256(raw).hexdigest(),analytic_control=ci==1,oracle_method='NumPyglobalaxis-cumsums, parametricray-planeintersection, tensorproductbilinearweights, finite-differencecells and sqrt rayanglefactors; no GPUstagecode copied',numpy=np.__version__))
(P/'fixture_provenance.json').write_text(json.dumps(dict(cases=allmeta,python=sys.version,generator='generate_oracle.py',atol=1e-10,rtol=1e-9),indent=2)+'\n');print([(x['case'],len(x['prefix']),x['oracle_count']) for x in allmeta])
