from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
HDT=f32(f32(.001)/f32(1/f32(math.pi)))

def layout(values,N,kid):
 if kid not in range(4,24):return [q for v in values for q in v]
 out=[0.]*(2*N*49)
 for n in range(N):
  for e in range(49):
   for c in range(2):out[n//4*392+8*e+4*c+n%4]=values[n*49+e][c]
 return out
f=[];o=[];cs=[]
for N,mode in[(256,0),(512,1),(256,2)]:
 H=[complex(f32(1-f32(e/49)),0)for e in range(49)]if mode==0 else [complex((i+1)/16 if i==j else 0,0)for i in range(7)for j in range(7)]if mode==1 else [complex(float(i==j),0)for i in range(7)for j in range(7)]
 sigma=[complex(f32(f32(n/N)-f32(e/49)),f32(f32(e/49)-f32(n/N)))for n in range(N)for e in range(49)]
 for kid in range(25):
  h=[complex(f32(v.real*HDT),f32(v.imag*HDT))for v in H]if kid>=3 else H
  ham=[q for v in h for q in[v.real,v.imag]]if kid not in range(4,24)else[v.real for v in h]+[v.imag for v in h]
  sin=layout([(v.real,v.imag)for v in sigma],N,kid);expected=[]
  if kid>0:
   for n in range(N):
    for i in range(7):
     for j in range(7):
      terms=[h[7*i+k]*sigma[n*49+7*k+j]-sigma[n*49+7*i+k]*h[7*k+j]for k in range(7)];comm=complex(math.fsum(v.real for v in terms),math.fsum(v.imag for v in terms));v=-1j*comm*(HDT if kid<3 else 1);expected.append((v.real,v.imag))
  out=layout(expected,N,kid)if kid>0 else []
  f +=[N,mode,kid,7,1001,HDT]+ham+sin;o+=out;cs.append({'N':N,'mode':mode,'kid':kid,'layout':'AoSoA4sigma/SoAH'if kid in range(4,24)else'AoScomplex','pre_scaled_H':kid>=3,'execution_only':kid==0,'output_count':len(out),'oracle_max':max(map(abs,out),default=0.),'trace_contract':'Commutatortrace0, matrixalgebra only; defaultHnonHermitian so noquantumdensitypositivity/Hermiticity claim','identity_H_requires_exact_zero':mode==2})
(p/'fixtures_f32.bin').write_bytes(struct.pack('<%sf'%len(f),*f));(p/'oracle_f64.bin').write_bytes(struct.pack('<%sd'%len(o),*o));(p/'independent_oracle.json').write_text(json.dumps({'method':'IndependentPythoncomplexmatrixcommutator withFloat64fsum fromactualpreparedFloat32H/sigma, source -i andhdt once, explicitstoragepermutation. DefaultHrealnonsymmetric, diagonalHanalyticdifference, identityHzero. SourceoriginalHimag0scope; notgeneralcomplexH validation.','HDT':HDT,'HDT_bits':struct.unpack('<I',struct.pack('<f',HDT))[0],'cases':cs,'input_prefix_count':len(f),'oracle_values':len(o),'acceptance':{'atol':1e-6,'rtol':1e-4,'identity_exact_zero':True,'before_GPU':True},'native_empty_contract':'kid0 unchangedemptykernel hasno numericaloutputs, executiononlytimingseparatecomponent; notmanufacturedoutputagreement'},indent=2)+'\n')
