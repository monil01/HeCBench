from pathlib import Path
import json,math,struct
R=Path('/home/imo/HeCBench_julia');p=R/'src/eikonal-julia'
f=[];out=[];cases=[]
def blockidx(x,y,z,n):return((z//4*(n//4)+y//4)*(n//4)+x//4)*64+(z%4)*16+(y%4)*4+x%4
def solve(n,speed):
 v=[1e20]*(n**3);v[0]=0
 def get(x,y,z):return v[max(0,min(n-1,x))+n*(max(0,min(n-1,y))+n*max(0,min(n-1,z)))]
 for it in range(1000):
  new=v.copy();delta=0
  for z in range(n):
   for y in range(n):
    for x in range(n):
     idx=x+n*(y+n*z)
     if idx==0:continue
     a=sorted([min(get(x-1,y,z),get(x+1,y,z)),min(get(x,y-1,z),get(x,y+1,z)),min(get(x,y,z-1),get(x,y,z+1))]);s=1/speed[blockidx(x,y,z,n)];u=a[0]+s
     for dim in[2,3]:
      if u<=a[dim-1]:break
      b=math.fsum(a[:dim]);c=math.fsum(t*t for t in a[:dim])-s*s;u=(b+math.sqrt(max(0,b*b-dim*c)))/dim
     new[idx]=min(v[idx],u);delta=max(delta,abs(new[idx]-v[idx]))
  v=new
  if delta<1e-13:break
 assert it<999
 return[v[x+n*(y+n*z)]for z0 in range(0,n,4)for y0 in range(0,n,4)for x0 in range(0,n,4)for z in range(z0,z0+4)for y in range(y0,y0+4)for x in range(x0,x0+4)],it+1
for n,iters,mode in[(4,1,0),(8,3,0),(12,10,1)]:
 speed=[1.+((x*3+y*5+z*7)%5)/4 if mode else 1. for z0 in range(0,n,4)for y0 in range(0,n,4)for x0 in range(0,n,4)for z in range(z0,z0+4)for y in range(y0,y0+4)for x in range(x0,x0+4)];init=[1e20]*n**3;init[0]=0.;ref,steps=solve(n,speed);f+=[n,iters,mode]+speed+init;out+=ref;cases.append({'n':n,'nIter':iters,'speed_mode':mode,'independent_global_Jacobi_iterations':steps,'oracle_max':max(ref)})
(p/'fixtures_f64.bin').write_bytes(struct.pack('<%sd'%len(f),*f));(p/'oracle_f64.bin').write_bytes(struct.pack('<%sd'%len(out),*out));(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent global monotone Jacobi with sorted-neighbor Godunov quadratic root solved in Float64 (not native blocked active-list algorithm); source seed0 and clamped/self boundary; convergence1e-13','cases':cases,'input_prefix_count':len(f),'outputs':len(out),'acceptance':'atol1e-5 rtol1e-5 fullsolution, plus finite/nonnegative seed0 and exactconstant-speed coordinate-axis travel times; native sqrtf differs from Float64 root'},indent=2)+'\n')
