import sys,numpy as np
m,n=500,1000
s=(m-2)*n; small=(m-2)*(n-2); count=11*s+small+3
a=np.fromfile(sys.argv[1],dtype=np.float32);b=np.fromfile(sys.argv[2],dtype=np.float32)
assert a.size==b.size==count,(a.size,b.size,count)
assert np.isfinite(a).all() and np.isfinite(b).all()
error=np.abs(a.astype(np.float64)-b.astype(np.float64))
assert np.allclose(a,b,rtol=1e-5,atol=1e-4),(np.argmax(error),error.max())
for name,x in [("native",a),("iris",b)]:
 before=x[:s];used_dt=x[s];fields=x[s+1:s+1+10*s].reshape(10,s)
 alpha=np.float32(used_dt/np.float32(4));alpha3=np.float32(np.float32(3)*alpha)
 stage=(before.astype(np.longdouble)+np.longdouble(alpha)*fields[1].astype(np.longdouble)).astype(np.float32)
 ref=(stage.astype(np.longdouble)+np.longdouble(alpha3)*fields[3].astype(np.longdouble)).astype(np.float32)
 delta=np.abs(fields[0].astype(np.float64)-ref.astype(np.float64))
 assert np.allclose(fields[0],ref,rtol=1e-6,atol=2e-7),(name,delta.max())
 print(f"PASS {name} independent final RK3 identity, all {s} temperature values, max_abs={delta.max()}")
 assert np.any(fields[1]!=0) and np.any(fields[3]!=0),"nontrivial RK3 stage derivatives required"
 print(f"{name} nontrivial stages: x_range=({fields[1].min()},{fields[1].max()}), z_range=({fields[3].min()},{fields[3].max()}), changed_temperature={np.count_nonzero(fields[0]!=before)}")
print(f"PASS native/IRIS all {count} finite float values: 10 state/work fields, prior-temperature field, omega, used/adaptive dt, Nusselt; max_abs={error.max()}, differing={np.count_nonzero(error)}")
