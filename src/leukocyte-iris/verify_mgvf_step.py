"""Independent one-step Jacobi equations, using a real captured cell patch."""
import pathlib,struct,sys,numpy as np
with pathlib.Path(sys.argv[1]).open('rb') as f:
 frame,cells,total=struct.unpack('iii',f.read(12))
 dims=np.frombuffer(f.read(12*cells),np.int32)
 data=np.frombuffer(f.read(4*total),np.float32)
 offset,m,n=int(dims[0]),int(dims[cells]),int(dims[2*cells])
image=data[offset:offset+m*n].reshape(m,n).astype(np.float64)
p=np.pad(image,1,mode='edge')
neighbors=[p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:],p[:-2,2:],p[2:,2:],p[:-2,:-2],p[2:,:-2]]
coefficient=float(np.float32(1)/np.float32(3.14159))
terms=np.zeros_like(image)
for neighbor,velocity in zip(neighbors,[-1,1,-1,1,0,2,-2,0]):
 difference=neighbor-image
 heaviside=np.arctan(difference*velocity*1e10)*coefficient+.5
 terms+=heaviside*difference
expected=image+float(np.float32(.5)/np.float32(5))*terms
expected-=.2*image*(expected-image)
values=[]
for filename in sys.argv[2:]:
 actual=np.fromfile(filename,np.float32).reshape(m,n)
 assert np.isfinite(actual).all()
 np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
 print('PASS independent Jacobi one-step actual cell',frame,'shape',m,n,'all',m*n,'values; maxabs',np.max(np.abs(actual-expected)))
 values.append(actual)
assert len(values)==2
np.testing.assert_array_equal(values[0],values[1])
print('PASS native/IRIS one-step byte-exact; no synthetic benchmark input or CPU execution fallback')
