"""Independent Gauss basis, SPD flux tensor and complete Helmholtz output witness."""
import pathlib,math,struct,json,hashlib
E=7;N=8
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def leg(x):
 p0,p1=1.,x
 for k in range(2,9):p0,p1=p1,((2*k-1)*x*p1-(k-1)*p0)/k
 return p1,8*(x*p1-p0)/(x*x-1)
xs=[];ws=[]
for j in range(8):
 x=math.cos(math.pi*(j+.75)/8.5)
 for _ in range(30):
  p,dp=leg(x);step=p/dp;x-=step
  if abs(step)<1e-15:break
 xs.append(x);ws.append(2/((1-x*x)*leg(x)[1]**2))
order=sorted(range(8),key=xs.__getitem__);xs=[xs[i]for i in order];ws=[ws[i]for i in order];bary=[1/math.prod(xs[i]-xs[j]for j in range(8)if j!=i)for i in range(8)];D=[]
for i in range(8):
 row=[bary[j]/(bary[i]*(xs[i]-xs[j]))if i!=j else 0 for j in range(8)];row[i]=-sum(row);D.extend(row)
D32=list(map(f32,D));polyerr=max(abs(sum(D32[i*8+j]*xs[j]**degree for j in range(8))-(degree*xs[i]**(degree-1)if degree else 0))for i in range(8)for degree in range(8));assert polyerr<1e-4
G=[0.]*(E*7*512);Q=[0.]*(3*E*512);LAM=[1.]*(E*512)+[1.125]*(E*512);tensor=[1,2,.125,.25,3,.375,4]
for e in range(E):
 for k in range(8):
  for j in range(8):
   for i in range(8):
    p=i+8*j+64*k;idx=e*512+p;weight=ws[i]*ws[j]*ws[k]*(1+e/16)
    for c,a in enumerate(tensor):G[e*7*512+c*512+p]=f32(weight*a)
    Q[idx]=f32(((37*p+19*e)%31-15)/32);Q[E*512+idx]=f32(xs[i]**2+xs[j]*xs[k]+.5*xs[i]*xs[j]+e/8);Q[2*E*512+idx]=1.25
# Independent direct tensor derivatives, weighted fluxes and transpose action.
OUT=[];energies=[];constant_errors=[]
for field in range(3):
 fieldout=[]
 for e in range(E):
  q=Q[field*E*512+e*512:field*E*512+(e+1)*512];flux=[[0.]*512 for _ in range(3)]
  for k in range(8):
   for j in range(8):
    for i in range(8):
     p=i+8*j+64*k;grad=[sum(D32[i*8+m]*q[m+8*j+64*k]for m in range(8)),sum(D32[j*8+m]*q[i+8*m+64*k]for m in range(8)),sum(D32[k*8+m]*q[i+8*j+64*m]for m in range(8))];g=[G[e*7*512+c*512+p]for c in range(7)];matrix=[[g[1],g[2],g[3]],[g[2],g[4],g[5]],[g[3],g[5],g[6]]]
     for axis in range(3):flux[axis][p]=sum(matrix[axis][v]*grad[v]for v in range(3))
  result=[]
  for k in range(8):
   for j in range(8):
    for i in range(8):
     p=i+8*j+64*k;mass=1.125*G[e*7*512+p]*q[p];value=mass+sum(D32[m*8+i]*flux[0][m+8*j+64*k]+D32[m*8+j]*flux[1][i+8*m+64*k]+D32[m*8+k]*flux[2][i+8*j+64*m]for m in range(8));result.append(value)
     if field==2:constant_errors.append(abs(value-mass))
  energies.append(sum(a*v for a,v in zip(q,result)));fieldout.extend(result)
 OUT.extend(fieldout)
assert min(energies)>0

import array,sys
result=json.loads(pathlib.Path(sys.argv[1]).read_text());expected=OUT[:E*512]+OUT;rows=[]
for model,record in result['models'].items():
 for trial in record['trials']:
  got=array.array('f');got.frombytes(pathlib.Path(trial['output_artifact']['saved_path']).read_bytes());assert len(got)==14336
  error=max(abs(a-b)for a,b in zip(got,expected));assert error<1e-3 and all(math.isfinite(v)for v in got)
  energies_gpu=[sum(Q[i]*got[i]for i in range(E*512))]+[sum(Q[field*E*512+i]*got[E*512+field*E*512+i]for i in range(E*512))for field in range(3)];assert min(energies_gpu)>0
  constant=max(abs(got[E*512+2*E*512+i]-1.125*G[(i//512)*7*512+i%512]*1.25)for i in range(E*512))
  rows.append({'model':model,'trial':trial['trial'],'all14336values_match_independent_Float64_tensor_oracle':True,'max_abs_error':error,'q_dot_Aq_per_case':energies_gpu,'bilinear_symmetry_abs_residual':abs(sum(Q[i]*got[2*E*512+i]for i in range(E*512))-sum(Q[E*512+i]*got[E*512+i]for i in range(E*512))),'constant_field_diffusion_residual_max_abs':constant})
assert len(rows)==20
print(json.dumps({'benchmark':'axhelm','trials':rows,'basis_Float32_max_polynomial_derivative_error_degrees0to7':polyerr,'quadrature_Float64_max_monomial_error_degrees0to15':max(abs(sum(w*x**degree for x,w in zip(xs,ws))-(2/(degree+1)if degree%2==0 else 0))for degree in range(16)),'minimum_independent_field_element_energy':min(energies),'Float64_oracle_constant_diffusion_rounding_max':max(constant_errors),'scope':'ElementlocalweightedSPDdiffusionplusmass,notglobalassembledPDEsolverconvergence/boundaryconditioncertificate'},indent=2))
