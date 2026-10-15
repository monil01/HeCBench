"""Whole-output double-precision projection oracle on the actual input dump."""
import pathlib,sys,numpy as np
native,iris=map(pathlib.Path,sys.argv[1:3]);batches=int(sys.argv[3]);total=0;worst=0.
rotation=np.array([[-.9111348390579224,.0751304179430008,-.4052018225193024],[-.3360927104949951,-.7044632434844971,.6251187324523926],[-.2384843230247498,.7057529091835022,.6671117544174194]],np.float32).astype(np.float64)
translation=np.array([.1,-.2,.3],np.float32).astype(np.float64)
d0,d1,d2,d3,d4=np.array([.03,-.15,.001,.001,.05],np.float32).astype(np.float64)
for batch in range(batches):
 arrays={}
 for field in ('input','x','y','z','valid'):
  dtype=np.int32 if field=='valid' else np.float32
  h=np.fromfile(native/f'batch_{batch}_{field}.bin',dtype);i=np.fromfile(iris/f'batch_{batch}_{field}.bin',dtype)
  assert h.size==(800000 if field=='input' else 100000) and np.isfinite(h).all() and np.isfinite(i).all()
  np.testing.assert_array_equal(h,i);arrays[field]=h;total+=h.size
 points=arrays['input'].reshape(-1,8)
 assert np.all((points[:,0:2]>=-1)&(points[:,0:2]<=1)) and np.all((points[:,2]>=15)&(points[:,2]<=25))
 # Independent vector affine projection, radial and tangential distortion.
 camera=points[:,:3].astype(np.float64)@rotation.T+translation
 valid=camera[:,2]>2.5;denom=np.where(valid,camera[:,2],1.)
 x,y=camera[:,0]/denom,camera[:,1]/denom;r2=x*x+y*y
 radial=1+d0*r2+d1*r2**2+d4*r2**3
 expected=[1200*(x*radial+2*d2*x*y+d3*(r2+2*x*x))+400.5,1200*(y*radial+d2*(r2+2*y*y)+2*d3*x*y)+300.5,camera[:,2]*100]
 np.testing.assert_array_equal(arrays['valid'],valid.astype(np.int32))
 for field,e in zip(('x','y','z'),expected):
  e=np.where(valid,e,0.);err=float(np.max(np.abs(arrays[field]-e)));worst=max(worst,err)
  np.testing.assert_allclose(arrays[field],e,rtol=0,atol=1e-3)
 print('PASS batch',batch,'all100000 points native/IRIS byte-exact; independent double projection maxabs <=',worst)
print('PASS all',total,'finite input/output entries; validity exact, original1e-3 absolute bound unchanged; maxabs',worst)
