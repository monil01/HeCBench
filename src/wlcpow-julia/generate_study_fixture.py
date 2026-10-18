from pathlib import Path
import struct,math,json
p=Path(__file__).resolve().parent;f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];bits=lambda x:struct.unpack('<i',struct.pack('<f',x))[0];mask=(1<<32)-1;pre=[5];gold=[];kinds=bytearray();cases=[];metrics=dict(rr_clipped=0,rlog_clipped=0,minimum_distance=1e100,minimum_u=1,maximum_gaussian=0)
def gauss(a,b,tes,see):
 pred=a>b;u=(a+tes)&mask;v=(b+see)&mask;v0,v1=(u,v)if pred else(v,u);s=0
 for _ in range(4):
  s=(s+0x9e3779b9)&mask;v0=(v0+((((v1<<4)+0xa341316c)&mask)^((v1+s)&mask)^(((v1>>5)+0xc8013ea4)&mask)))&mask;v1=(v1+((((v0<<4)+0xad90777d)&mask)^((v0+s)&mask)^(((v0>>5)+0x7e95761e)&mask)))&mask
 signed=v0 if v0<2**31 else v0-2**32;x=f32(v1)*2**-32;angle=f32(signed)*2**-31;assert x>0;metrics['minimum_u']=min(metrics['minimum_u'],x);value=max(-4,min(4,math.sqrt(-2*math.log(x))*math.sin(math.pi*angle)));metrics['maximum_gaussian']=max(metrics['maximum_gaussian'],abs(value));return value
for ci,(N,nt,reps)in enumerate([(2,0,1),(127,1,2),(128,4,3),(129,32,4),(513,3,2)]):
 period=[4.,4.,4.]if ci==0 else[16.,16.,16.];C=[];V=[]
 for i in range(N+1):
  xyz=[0.,0.,0.]if ci==0 and i==0 else[1.,.5,.25]if ci==0 and i==1 else[i/64,(i%7)/16,(i%11)/32]
  vel=[((i*3)%7-3)/16,((i*5)%11-5)/32,((i*7)%13-6)/32];w=f32(math.sqrt(sum(x*x for x in vel)));C.extend(map(f32,xyz+[0.]));V.extend(map(f32,vel+[w]))
 nb=[(1 if ci==0 else 0 if i%4==3 else 1+i%3)if i<N else 0 for i in range(N+1)];bonds=[];l0=[]
 for s in range(2*N+1):bonds.extend([1-s if ci==0 and s<2 else(s+1)%(N+1),0 if ci==0 else s%(nt+1)]);l0.append(1+(s%5)/8+2**-30)
 params=[]
 for tab in range(8):
  for t in range(nt+1):
   temp=f32(.75+t/128);gamt=f32(.125+t/1024);gamc=f32(4*gamt)
   vals=[temp,f32(2+t/64),f32(.5+t/256),f32(1.25+(t%3)/4),gamc,gamt,0. if ci==0 else f32(math.sqrt(2*temp*(3*gamc-gamt))),0. if ci==0 else f32(2*math.sqrt(gamt*temp))];params.append(vals[tab])
 initial=[[((i*(axis+3))%17-8)/16 for i in range(N+1)]for axis in range(3)];forces=[x.copy()for x in initial];inc=[[0.]*(N+1)for _ in range(3)]
 for i in range(N):
  x=C[4*i:4*i+3];v=V[4*i:4*i+3]
  for bond in range(nb[i]):
   s=i+bond;j,typ=bonds[2*s:2*s+2];temp,r0,mu,q,gc,gt,sc,st=[params[tab*(nt+1)+typ]for tab in range(8)];de=[]
   for a in range(3):
    d=x[a]-C[4*j+a];h=period[a]/2;de.append(d+(0 if d>-h and d<h else-period[a]if d>-h else period[a]))
   dv=[v[a]-V[4*j+a]for a in range(3)];ra=math.hypot(*de);assert ra>0;metrics['minimum_distance']=min(metrics['minimum_distance'],ra);ell=f32(l0[s]);lm=ell*r0;rr=1/r0;sr=(1-rr)**2;kph=ell**q*temp*(.25/sr-.25+rr);mod=.433*(temp*(-.25/sr+.25+.5*rr/(sr*(1-rr)))/(lm*rr)+kph*(q+1)/(ell**(q+1)));lam=mod/mu;kph/=lam;ratio=ra/lm;power=ra**(q+1);metrics['rr_clipped']+=ratio>=.99;metrics['rlog_clipped']+=power<.01;ratio=min(.99,ratio);power=max(.01,power);vv=sum(a*b for a,b in zip(de,dv))/ra;w=[[gauss(bits(V[4*i+3]),bits(V[4*j+3]),a,b)for b in range(3)]for a in range(3)];trace=sum(w[a][a]for a in range(3))/3;wr=[sum(((w[a][b]-trace)if a==b else .5*(w[a][b]+w[b][a]))*de[b]for b in range(3))for a in range(3)];ff=-temp*(.25/(1-ratio)**2-.25+ratio)/lam/ra+kph/power+(sc*trace-gc*vv)/ra
   for a in range(3):inc[a][i]+=de[a]*ff-gt*dv[a]+st*wr[a]/ra
 for a in range(3):
  for i in range(N+1):forces[a][i]+=reps*inc[a][i]
 pre.extend([N,nt,reps,*period]);arrays=[(C,4),(V,4),(nb,1),(bonds,2),(l0,1)]+[(params[t*(nt+1):(t+1)*(nt+1)],1)for t in range(8)];pre.extend(C);pre.extend(V);pre.extend(nb);pre.extend(bonds);pre.extend(l0);pre.extend(params)
 for x in initial:pre.extend(x)
 def append(values,width=1,exact=True):
  g=[16384]*(64*width);gold.extend(g+values+g);kinds.extend(bytes([1])*(64*width));kinds.extend(bytes([1 if exact else 0])*len(values));kinds.extend(bytes([1])*(64*width))
 for x in initial:append(x)
 for a,x in enumerate(forces):
  append(x,exact=False)
  start=len(kinds)-64-len(x)
  for i in range(N+1):
   if i==N or nb[i]==0:kinds[start+i]=1
 for values,w in arrays:append(values,w)
 cases.append(dict(N=N,n_type=nt,repeats=reps,padding=1,period=period,neighbors=nb,nonzero_initial_forces=True,params_source_domain='positivefiniteWLCratior0>1,nonsingularlambda,distance>0; notoriginalrandomCLIphysicsscope'))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'oracle.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));(p/'kinds.bin').write_bytes(kinds);(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_count=len(gold),artifact_count=len(pre)+len(gold),independent_oracle='Float64analyticWLC/power/viscousforce+exactwrapped4roundTEA+BoxMullerlog/sin (notnativefloatlog2/sinpif), sourceFloat32randomcast andl0cast retained; engineeringbudgetnotformalphysicalvalidation',metrics=metrics,initial_force_and_all_readonly_inputs_exact=True,all_zero_neighbor_and_ghost_forces_exact=True,reciprocal_noiseoff_pair_case=0),indent=2)+'\n');print(len(pre),len(gold),metrics)
