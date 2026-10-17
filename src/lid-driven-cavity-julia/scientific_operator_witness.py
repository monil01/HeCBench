#!/usr/bin/env python3
"""Independent full physical-grid projection oracle, not packed pressure indices.
Named N32, end time .02 exercises three transient steps, including advection.
"""
import math,struct,json,argparse,hashlib
from pathlib import Path
N=32;B=16;H=N//2;DX=1/N;RE=1000.;OMEGA=1.7;GAMMA=.9;TAU=.5;END=.02;TOL=.001

def zeros(n=N+2):return [[0. for _ in range(n)] for _ in range(n)]
def flatten(a):return [x for row in a for x in row]
def host_bc(u,v):
 for i in range(N+2):
  u[0][i]=0.;v[0][i]=-v[1][i];u[N][i]=0.;v[N+1][i]=-v[N][i];u[i][0]=-u[i][1];v[i][0]=0.;u[i][N+1]=2-u[i][N];v[i][N]=0.
  if i==N:
   u[0][0]=0.;v[0][0]=-v[1][0];u[0][N+1]=0.;v[0][N+1]=-v[1][N+1];u[N][0]=0.;v[N+1][0]=-v[N][0];u[N][N+1]=0.;v[N+1][N+1]=-v[N][N+1]
   u[0][0]=-u[0][1];v[0][0]=0.;u[N+1][0]=-u[N+1][1];v[N+1][0]=0.;u[0][N+1]=2-u[0][N];v[0][N]=0.;u[N+1][N+1]=2-u[N+1][N];v[i][N+1]=0.
def device_bc(u,v):
 # Noncorner lanes then the source's last-lane corner overwrite order.
 for i in range(1,N+1):
  u[0][i]=0.;v[0][i]=-v[1][i];u[N][i]=0.;v[N+1][i]=-v[N][i];u[i][0]=-u[i][1];v[i][0]=0.;u[i][N+1]=2-u[i][N];v[i][N]=0.
 u[N][N+1]=0.;u[0][0]=-u[0][1];v[0][0]=0.;u[N+1][0]=-u[N+1][1];v[N+1][0]=0.;u[0][N+1]=2-u[0][N];v[0][N]=0.;u[N+1][N+1]=2-u[N+1][N];v[N][N+1]=0.;v[0][N+1]=-v[1][N+1];v[N+1][N+1]=-v[N][N+1]
def packed(p):
 pr=[[0.]*(H+2) for _ in range(N+2)];pb=[[0.]*(H+2) for _ in range(N+2)]
 for i in range(1,N+1):
  for j in range(1,H+1):pr[i][j]=p[i][2*j-(i&1)];pb[i][j]=p[i][2*j-((i+1)&1)]
 for i in range(1,N+1,2):pb[i][0]=p[i][0];pr[i+1][0]=p[i+1][0];pr[i][H+1]=p[i][N+1];pb[i+1][H+1]=p[i+1][N+1]
 for j in range(1,H+1):pr[0][j]=p[0][2*j];pb[0][j]=p[0][2*j-1];pr[N+1][j]=p[N+1][2*j-1];pb[N+1][j]=p[N+1][2*j]
 return pr,pb

def oracle():
 u=zeros();v=zeros();F=zeros();G=zeros();p=zeros();host_bc(u,v);zpres=[0.]*(34*18)
 prefix=[N,B,RE,OMEGA,GAMMA,TAU,END,TOL]+flatten(u)+flatten(v)+flatten(F)+flatten(G)+zpres+zpres
 maxu=max(abs(x) for row in u for x in row);maxv=1e-10;t=0.;records=[]
 while t<END:
  dt=min(TAU*min(.5*RE/(2/DX**2),DX/maxu,DX/maxv),END-t)
  for i in range(1,N):
   for j in range(1,N+1):
    a=u[i][j];r=u[i+1][j];l=u[i-1][j];top=u[i][j+1];bot=u[i][j-1]
    du2=((a+r)**2-(l+a)**2+GAMMA*(abs(a+r)*(a-r)-abs(l+a)*(l-a)))/(4*DX)
    uv=((v[i][j]+v[i+1][j])*(a+top)-(v[i][j-1]+v[i+1][j-1])*(bot+a)+GAMMA*(abs(v[i][j]+v[i+1][j])*(a-top)-abs(v[i][j-1]+v[i+1][j-1])*(bot-a)))/(4*DX)
    F[i][j]=a+dt*((r-2*a+l+top-2*a+bot)/(DX**2*RE)-du2-uv)
  for j in range(1,N+1):F[0][j]=u[0][j];F[N][j]=u[N][j]
  for i in range(1,N+1):
   for j in range(1,N):
    a=v[i][j];r=v[i+1][j];l=v[i-1][j];top=v[i][j+1];bot=v[i][j-1]
    dv2=((a+top)**2-(bot+a)**2+GAMMA*(abs(a+top)*(a-top)-abs(bot+a)*(bot-a)))/(4*DX)
    uv=((u[i][j]+u[i][j+1])*(a+r)-(u[i-1][j]+u[i-1][j+1])*(l+a)+GAMMA*(abs(u[i][j]+u[i][j+1])*(a-r)-abs(u[i-1][j]+u[i-1][j+1])*(l-a)))/(4*DX)
    G[i][j]=a+dt*((r-2*a+l+top-2*a+bot)/(DX**2*RE)-dv2-uv)
  for i in range(1,N+1):G[i][0]=v[i][0];G[i][N]=v[i][N]
  ps=[sum(p[i][j]**2 for j in range(1,N+1)) for i in range(1,N+1)];p0=math.sqrt(sum(ps)/(N*N));p0=1. if p0<.0001 else p0
  rhs=zeros()
  for i in range(1,N+1):
   for j in range(1,N+1):rhs[i][j]=((F[i][j]-F[i-1][j])/DX+(G[i][j]-G[i][j-1])/DX)/dt
  for iteration in range(1,1000001):
   for i in range(1,N+1):p[i][0]=p[i][1];p[i][N+1]=p[i][N]
   for j in range(1,N+1):p[0][j]=p[1][j];p[N+1][j]=p[N][j]
   for color in (0,1):
    for i in range(1,N+1):
     for j in range(1,N+1):
      if (i+j)%2==color:p[i][j]=p[i][j]*(1-OMEGA)+OMEGA*((p[i-1][j]+p[i+1][j]+p[i][j-1]+p[i][j+1])/DX**2-rhs[i][j])/(4/DX**2)
   res=[sum(((p[i-1][j]-2*p[i][j]+p[i+1][j])/DX**2+(p[i][j-1]-2*p[i][j]+p[i][j+1])/DX**2-rhs[i][j])**2 for j in range(1,N+1)) for i in range(1,N+1)]
   norm=math.sqrt(sum(res)/(N*N))/p0
   if norm<TOL:break
  else:raise RuntimeError('CPU SOR did not converge')
  for i in range(1,N):
   for j in range(1,N+1):u[i][j]=F[i][j]-dt*(p[i+1][j]-p[i][j])/DX
  for i in range(1,N+1):
   for j in range(1,N):v[i][j]=G[i][j]-dt*(p[i][j+1]-p[i][j])/DX
  # Source reduction includes old ghost values before the final BC application.
  mu=[];mv=[]
  for i in range(1,N+1):
   vals=[]
   for r in range(1,H+1):
    if i!=N:
     a=max(abs(u[i][2*r-(i&1)]),abs(u[i][2*r-((i+1)&1)]))
     if 2*r==N:a=max(a,abs(u[i][N+1]))
    else:a=max(abs(u[N][2*r]),abs(u[0][2*r]),abs(u[N][2*r-1]),abs(u[0][2*r-1]),abs(u[N+1][2*r]),abs(u[N+1][2*r-1]))
    vals.append(a)
   mu.append(max(vals));vals=[]
   for r in range(1,H+1):
    if r!=H:
     a=max(abs(v[i][2*r-(i&1)]),abs(v[i][2*r-((i+1)&1)]))
     if i==N:a=max(a,abs(v[N+1][2*r]))
    else:
     j=N-1;a=max(abs(v[i][j]),abs(v[i][N]),abs(v[i][0]),abs(v[i][N+1]))
    vals.append(a)
   mv.append(max(vals))
  maxu=max(1e-10,*mu);maxv=max(1e-10,*mv);device_bc(u,v);t+=dt
  divergence=math.sqrt(sum(((u[i][j]-u[i-1][j])/DX+(v[i][j]-v[i][j-1])/DX)**2 for i in range(1,N+1) for j in range(1,N+1))/(N*N))
  fresh_residual=math.sqrt(sum(((p[max(i-1,1)][j]-2*p[i][j]+p[min(i+1,N)][j])/DX**2+(p[i][max(j-1,1)]-2*p[i][j]+p[i][min(j+1,N)])/DX**2-rhs[i][j])**2 for i in range(1,N+1) for j in range(1,N+1))/(N*N))
  assert abs(divergence-dt*fresh_residual)<1e-12
  records.append({'time':t,'dt':dt,'sor_iterations':iteration,'relative_residual':norm,'pressure_scale':p0,'divergence_rms':divergence,'fresh_Neumann_residual_rms':fresh_residual,'source_stale_ghost_residual_rms':norm*p0,'discrete_projection_identity_error':abs(divergence-dt*fresh_residual)})
 pr,pb=packed(p);out=flatten(u)+flatten(v)+flatten(F)+flatten(G)+flatten(pr)+flatten(pb)+ps+res+mu+mv+[iteration,dt,norm,len(records)]
 assert len(prefix)==5856 and len(out)==5980
 return prefix,out,records

def main():
 a=argparse.ArgumentParser();a.add_argument('--generate',action='store_true');a.add_argument('--result',type=Path);args=a.parse_args();prefix,out,records=oracle();p=Path(__file__).parent
 if args.generate:
  for name,values in [('study_input_prefix.bin',prefix),('study_expected.bin',out)]:
   (p/name).write_bytes(struct.pack('<%dd'%len(values),*values))
  meta={'named_grid':N,'block_size':B,'time_end':END,'source_time_end':.001,'reason':'Named longer transient exercises advection after the rest-state first projection; physical parameter change shared by all4, chosen before GPU execution, no source-kernel modification','prefix_count':len(prefix),'defined_state_and_partial_control_count':len(out),'records':records,'files':{name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['study_input_prefix.bin','study_expected.bin']}}
  (p/'scientific_fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
 if args.result:
  result=json.loads(args.result.read_text());expected=prefix+out;rows=[]
  for model,item in result['models'].items():
   for row in item['trials']:
    ar=row['output_artifact'];raw=Path(ar['saved_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ar['sha256'];values=struct.unpack('<%dd'%(len(raw)//8),raw);assert values[:5856]==tuple(prefix);err=max(abs(x-y) for x,y in zip(values,expected));assert len(values)==11836 and all(abs(x-y)<=1e-9+1e-10*abs(y) for x,y in zip(values,expected));rows.append({'model':model,'trial':row['trial'],'maximum_absolute_error':err})
  print(json.dumps({'records':records,'artifacts':rows},indent=2))
if __name__=='__main__':main()
