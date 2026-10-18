
from pathlib import Path
import struct,json,hashlib,sys,math
P=Path(__file__).resolve().parent
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
ALPHA=f32(.000122);BETA=.75;SIZE=5
DIMS=[(2,7,3,4,5),(1,4,1,3,7)]
def forward(x,dims):
 n,c,d,h,w=dims;plane=d*h*w;out=[]
 for i,a in enumerate(x):
  batch=i//(c*plane);channel=i//plane%c;p=i%plane
  sq=sum(x[batch*c*plane+j*plane+p]**2 for j in range(max(0,channel-2),min(c,channel+3)))
  out.append(a*(1+ALPHA/SIZE*sq)**(-BETA))
 return out
def gradient(x,g,dims):
 n,c,d,h,w=dims;plane=d*h*w;omega=[]
 for i in range(len(x)):
  batch=i//(c*plane);channel=i//plane%c;p=i%plane
  omega.append(1+ALPHA/SIZE*sum(x[batch*c*plane+j*plane+p]**2 for j in range(max(0,channel-2),min(c,channel+3))))
 out=[]
 for i,a in enumerate(x):
  batch=i//(c*plane);channel=i//plane%c;p=i%plane
  out.append(g[i]*omega[i]**(-BETA)-2*ALPHA*BETA/SIZE*a*sum(x[j]*g[j]*omega[j]**(-BETA-1)for j in [batch*c*plane+k*plane+p for k in range(max(0,channel-2),min(c,channel+3))]))
 return out
def main():
 files={};cases=[]
 for case,dims in enumerate(DIMS,1):
  count=math.prod(dims);src=[((i*37%101)+1)/128 for i in range(count)]if case==1 else[((i*17%43)-21)/16 for i in range(count)]
  g=list(src)if case==1 else[((i*13%31)-15)/32 for i in range(count)]
  fwd=forward(src,dims);grad=gradient(src,g,dims)
  # Independent numerical derivative of scalar dot(g,forward(x)), all coordinates.
  errors=[]
  for j in range(count):
   step=1e-4*max(1,abs(src[j]));xp=list(src);xm=list(src);xp[j]+=step;xm[j]-=step
   lp=math.fsum(a*b for a,b in zip(g,forward(xp,dims)));lm=math.fsum(a*b for a,b in zip(g,forward(xm,dims)))
   errors.append(abs((lp-lm)/(2*step)-grad[j]))
  assert max(errors)<1e-8
  for name,fmt,v in[(f'study_case{case}_f32.bin','f',src+g),(f'oracle_case{case}_f64.bin','d',fwd+grad)]:
   data=struct.pack('<'+fmt*len(v),*v);(P/name).write_bytes(data);files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  cases.append(dict(case=case,dims=list(dims),count=count,upstream_equals_src=case==1,finite_difference_coordinates=count,max_abs_analytic_vs_finite_difference_gradient=max(errors),source_depth_indexing_witness=case==1,integer_omega_sum_witness=case==2))
 (P/'fixture_provenance.json').write_text(json.dumps(dict(cases=cases,files=files,alpha=ALPHA,beta=BETA,k=1.,window=5,oracle='IndependentFloat64channel-window forwardpower andanalyticvector-Jacobian gradient; completecentraldifferenceofscalarweightedloss all924coordinates',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version),indent=2)+'\n')
if __name__=='__main__':main()
