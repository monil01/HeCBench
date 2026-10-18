"""Independent conservative Euler flux vectors, then radius-four centered divergence."""
from pathlib import Path
import array,struct,json,math,hashlib
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
c=list(map(f32,[.8,-.2,.038,-.0035]));dx=list(map(f32,[.01,.02,.03]));moment=2*math.fsum((r+1)*v for r,v in enumerate(c));fixture=array.array('d');oracle=array.array('d');notes=[]
for mode,n in enumerate([12,17,24]):
 v=n**3;fields=[[]for _ in range(8)]
 for k in range(n):
  for j in range(n):
   for i in range(n):
    if mode==0:rho=1.;u=.25;vv=-.125;w=.0625;press=.75
    elif mode==1:rho=1+(i+2*j+3*k)/128;u=.25;vv=-.125;w=.0625;press=.75+(2*i+j+k)/512
    else:rho=1+(i*i+2*j+3*k)/2048;u=.125+(j-4)/1024;vv=-.0625+(k-3)/2048;w=.03125+(i-2)/2048;press=.5+(i+2*j+k)/4096
    vals=[rho*u,rho*vv,rho*w,2*press+.5*rho*(u*u+vv*vv+w*w),u,vv,w,press]
    for a,x in zip(fields,vals):a.append(x)
 # Build physical flux vectors independently of native assignment expressions.
 vectors=[]
 for axis in range(3):
  F=[[]for _ in range(5)]
  for t in range(v):
   ua=fields[4+axis][t];press=fields[7][t]
   vals=[fields[axis][t]]+[fields[q][t]*ua+(press if q==axis else 0.)for q in range(3)]+[(fields[3][t]+press)*ua]
   for a,x in zip(F,vals):a.append(x)
  vectors.append(F)
 out=[[0.]*v for _ in range(5)]
 for k in range(4,n-4):
  for j in range(4,n-4):
   for i in range(4,n-4):
    t=(k*n+j)*n+i
    for q in range(5):out[q][t]=-math.fsum(dx[a]*math.fsum(c[r-1]*(vectors[a][q][t+r*stride]-vectors[a][q][t-r*stride])for r in range(1,5))for a,stride in enumerate([1,n,n*n]))
 affine_error=None;continuum_error=None
 if mode==0:assert all(x==0 for a in out for x in a)
 if mode==1:
  vel=[.25,-.125,.0625];gr=[1/128,2/128,3/128];gp=[2/512,1/512,1/512];vel2=sum(x*x for x in vel);analytic=[]
  for q in range(5):
   partial=[gr[a]*vel[a] if q==0 else (gr[a]*vel[q-1]*vel[a]+(gp[a]if q-1==a else 0.))if q<4 else (3*gp[a]+.5*vel2*gr[a])*vel[a]for a in range(3)]
   analytic.append(-math.fsum(dx[a]*partial[a]for a in range(3)))
  affine_error=max(abs(out[q][(k*n+j)*n+i]-moment*analytic[q])for q in range(5)for k in range(4,n-4)for j in range(4,n-4)for i in range(4,n-4));continuum_error=max(abs((moment-1)*x)for x in analytic);assert affine_error<1e-12
 fixture.extend([n,mode,10,*dx])
 for a in fields:fixture.extend(a)
 for a in out:oracle.extend(a)
 notes.append({'N':n,'mode':mode,'volume':v,'positive_density_pressure':True,'gamma':1.5,'manufactured_state':['constant','affine_rho_pressure_constant_velocity','varying_density_velocity_pressure'][mode],'affine_actual_stencil_first_moment_error':affine_error,'affine_continuum_coefficient_rounding_error':continuum_error,'max_abs_oracle_output':max(abs(x)for a in out for x in a)})
(p/'fixtures_f64.bin').write_bytes(fixture.tobytes());(p/'oracle_f64.bin').write_bytes(oracle.tobytes());(p/'independent_oracle.json').write_text(json.dumps({'method':__doc__,'source_Float32_coefficients_promoted_to_Float64':c,'source_dxinv_Float32_promoted_to_Float64':dx,'actual_stencil_first_moment':moment,'ideal_first_moment':1,'no_eighth_order_certificate':True,'atol':1e-12,'rtol':1e-10,'constant_requires_exact_zero':True,'halo_requires_exact_zero':True,'input_prefix_count':len(fixture),'oracle_output_count':len(oracle),'cases':notes,'assets':{'fixture_sha256':hashlib.sha256(fixture.tobytes()).hexdigest(),'oracle_sha256':hashlib.sha256(oracle.tobytes()).hexdigest()}},indent=2)+'\n');print({'prefix':len(fixture),'outputs':len(oracle),'first_moment':moment,'cases':notes})
