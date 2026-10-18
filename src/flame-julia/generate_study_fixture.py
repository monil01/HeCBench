"""Independent mathematical position oracle with true Float16 state, strict Float32 color recurrence."""
from pathlib import Path
import struct,math,random,json
p=Path(__file__).resolve().parent;N=256;NP=64;IT=15;NR=1024;PERM=128
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
f16=lambda x:struct.unpack('<e',struct.pack('<e',x))[0]
seed=(p/'params_original_default.bin').read_bytes();assert len(seed)==1928
weights=[f32(1),f32(.5),f32(.6)]+[0.]*17;s=0.
for w in weights:s=f32(s+w)
mapping=[int(f32(f32(w/s)*N)) for w in weights];mapping[0]+=N-sum(mapping)
for i in range(1,20):mapping[i]+=mapping[i-1]
params=[]
for case in range(2):
 b=bytearray(seed);struct.pack_into('<20i',b,1848,*mapping)
 if case==1:
  for j in range(20):
   for k in range(5):struct.pack_into('<if',b,(j*5+k)*8,0,1. if k==0 else 0.)
   struct.pack_into('<6f',b,880+j*24,1,0,0,0,1,0)
 params.append(bytes(b));p.joinpath('params_default.bin' if case==0 else 'params_linear.bin').write_bytes(b)
def radical(n):
 out=0.;weight=.5
 while n:out+=(n%2)*weight;n//=2;weight*=.5
 return out
start=[(f32((i/N-.5)*2),f32((radical(i)-.5)*2)) for i in range(N)]
randoms=[f32(((i*97+13)%1024)/1024) for i in range(NR)];randoms[0]=.5;randoms[1]=.75;randoms[2]=.25
rng=random.Random(9731);perms=[]
for j in range(PERM):
 row=list(range(N));rng.shuffle(row);assert sorted(row)==list(range(N));perms+=row
p.joinpath('start.bin').write_bytes(struct.pack('<%sf'%(2*N),*(x for xy in start for x in xy)));p.joinpath('random.bin').write_bytes(struct.pack('<%sf'%NR,*randoms));p.joinpath('permutations.bin').write_bytes(struct.pack('<%sH'%len(perms),*perms))
words=[list(struct.unpack('<482I',b)) for b in params];fixture=[N,NP,IT,NR,PERM,2]+words[0]+words[1]+[x for xy in start for x in xy]+randoms+perms
p.joinpath('fixture.bin').write_bytes(struct.pack('<%sd'%len(fixture),*fixture))
gold=[];diagnostics=[]
for case,b in enumerate(params):
 funcs=[];color=struct.unpack_from('<20f',b,800);pre=struct.unpack_from('<120f',b,880);post=struct.unpack_from('<120f',b,1360);variations=[struct.unpack_from('<if',b,k*8) for k in range(100)]
 for i in range(N):
  idx=i&~31;func=next(j for j,v in enumerate(mapping) if v>idx);funcs.append(func)
 domain={'min_radius':float('inf'),'zero_radius_count':0,'finite_iteration_calls':0,'min_negative_y_branch_abs_x':float('inf')}
 def iteration(x,y,c,i):
  j=funcs[i];x/=2;y/=2
  if i%3==1:x+=.5
  elif i%3==2:y+=.5
  a=pre[j*6:j*6+6];x,y=a[0]*x+a[1]*y+a[2],a[3]*x+a[4]*y+a[5]
  r=math.hypot(x,y);theta=math.atan2(x,y);domain['min_radius']=min(domain['min_radius'],r);domain['zero_radius_count']+=r==0
  if y<0:domain['min_negative_y_branch_abs_x']=min(domain['min_negative_y_branch_abs_x'],abs(x))
  ox=oy=0.
  for vid,factor in variations[j*5:j*5+5]:
   if abs(factor)<.01:continue
   if vid==0:u,v=x,y
   elif vid==5:u,v=theta/f32(3.1415926535),r-1
   elif vid==7:u,v=r*math.sin(theta*r),-r*math.cos(theta*r)
   elif vid==9:u,v=r*(math.cos(theta)+math.sin(r)),r*(math.sin(theta)-math.cos(r))
   elif vid==13:
    omega=0. if randoms[i]>.5 else f32(3.1415926535);arg=theta/2+omega;u,v=math.sqrt(r)*math.cos(arg),math.sqrt(r)*math.sin(arg)
   else:raise ValueError('Outside selected variation contract')
   ox+=factor*u;oy+=factor*v
  a=post[j*6:j*6+6];x,y=a[0]*ox+a[1]*oy+a[2],a[3]*ox+a[4]*oy+a[5]
  c=f32(f32(c+color[j])/2);assert math.isfinite(x) and math.isfinite(y) and math.isfinite(c);domain['finite_iteration_calls']+=1
  return x,y,c
 points=[None]*N;colors=[None]*N
 for i in range(N):
  xy=start[perms[i]];x,y,c=iteration(*xy,.5,i);points[i]=(f16(x),f16(y));colors[i]=f16(c)
 for step in range(1,16):
  for i in range(N):
   q=perms[step*N+i];x,y,c=iteration(*points[q],colors[q],i);points[q]=(f16(x),f16(y));colors[q]=f16(c)
 vertices=[]
 for step in range(64):
  for i in range(N):
   q=perms[(16+step)*N+i];vertices+=list(iteration(*points[q],colors[q],i))
 output=[-777.]*16+[v for xy in points for v in xy]+[-777.]*16+[-777.]*8+colors+[-777.]*8+[-777.]*24+vertices+[-777.]*24
 gold+=output;diagnostics.append(dict(case=case,name='source_default_four_variations' if case==0 else 'linear_dyadic_operator_control',function_counts={str(j):funcs.count(j) for j in set(funcs)},**domain))
p.joinpath('oracle.bin').write_bytes(struct.pack('<%sd'%len(gold),*gold));p.joinpath('fixture_description.json').write_text(json.dumps(dict(N=N,num_points_per_thread=NP,num_iterations=IT,num_randoms=NR,num_permutations=PERM,case_count=2,input_prefix_count=len(fixture),output_count=len(gold),oracle='Float64 nonlinear position mathematics with sourceFloat32 constants and trueFloat16statequantization; exactFloat32color recurrence. Linearcontrol binarydyadic positions expectedexact. Thisisnot universalhighprecisionerrorbound/rendererqualityproof.',position_atol=2**-9,position_rtol=2**-9,contract_basis='Predeclared engineeringbudget scaledtohalf-storage resolution, notprovenlong-timechaoticerrorbound; allfullpoints mustpass bothrefsandindependentoracle, no postfailuretoleranceadjustment. Exactcolors/guards/inputbits; linearcontrolcoordinateequalitysecondary.',native_arithmetic='defaultnvcc-O3, fmaddefaultenabled, no-fmadfalseflag',state_domain=diagnostics),indent=2,allow_nan=False))
