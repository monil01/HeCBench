from pathlib import Path
import math,struct,json
p=Path(__file__).resolve().parent
cases=[(16,1,0),(16,3,0),(24,3,1),(16,3,2)];fixtures=[];expected=[]
eps=.07;dx=.4;dt=.01;D=2.;delta=.8;a1=1.25/math.sqrt(2);lam=a1/.5;tau=a1*.64/(.5*D)
for n,steps,mode in cases:
 v=n**3;ph=[];u=[]
 for x in range(n):
  for y in range(n):
   for z in range(n):
    r=math.sqrt((x-.5*n)**2+(y-.5*n)**2+(z-.5*n)**2)
    ph.append(1. if r<5 else -1.);u.append(0. if r<5 else -.8*(1-math.exp(-(r-5))))
    if mode==1:ph[-1]=((17*x+11*y+7*z)%31-15)/32;u[-1]=((13*x+3*y+19*z)%29-14)/64
    if mode==2:ph[-1]=-1.;u[-1]=-.8
 fixtures.extend(ph+u)
 pn=[-8192.25]*v;un=pn.copy();forces=[[0.]*v for _ in range(3)]
 def grad(a,i):return [(a[i+s]-a[i-s])/(2*dx) for s in [n*n,n,1]]
 for step in range(steps):
  forces=[[0.]*v for _ in range(3)]
  for x in range(1,n-1):
   for y in range(1,n-1):
    for z in range(1,n-1):
     i=(x*n+y)*n+z;g=grad(ph,i);q=sum(t*t for t in g);sum4=sum(t**4 for t in g);A=1-5*eps/3 if q==0 else 1-3*eps+4*eps*sum4/(q*q)
     for k in range(3):
      derivative=0. if q==0 else g[k]*(g[k]*g[k]*q-sum4)/(q*q)
      forces[k][i]=A*A*g[k]+q*A*16*eps*derivative
  for x in range(n):
   for y in range(n):
    for z in range(n):
     i=(x*n+y)*n+z
     if min(x,y,z)==0 or max(x,y,z)==n-1:pn[i]=-1.;continue
     g=grad(ph,i);q=sum(t*t for t in g);A=1-5*eps/3 if q==0 else 1-3*eps+4*eps*sum(t**4 for t in g)/(q*q)
     div=sum((forces[k][i+s]-forces[k][i-s])/(2*dx) for k,s in enumerate([n*n,n,1]));s=1-ph[i]*ph[i];reaction=-ph[i]*s+lam*u[i]*s*s;pn[i]=ph[i]+dt/(tau*A*A)*(div-reaction)
  for x in range(n):
   for y in range(n):
    for z in range(n):
     i=(x*n+y)*n+z
     if min(x,y,z)==0 or max(x,y,z)==n-1:un[i]=-.8;continue
     lap=sum(u[i+s]+u[i-s]-2*u[i] for s in [n*n,n,1])/(dx*dx);un[i]=u[i]+.5*(pn[i]-ph[i])+dt*D*lap
  ph,pn=pn,ph;u,un=un,u
 expected.extend(ph+pn+u+un+sum(forces,[]))
for name,data in [('fixtures_f64.bin',fixtures),('oracle_f64.bin',expected)]:
 with(p/name).open('wb')as f:f.write(struct.pack('<'+'d'*len(data),*data))
(p/'study_cases.json').write_text(json.dumps(cases))
