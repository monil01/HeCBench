from pathlib import Path
import json,math,struct,hashlib
B=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
N=4099
v=list(map(f32,[-.3,-.6,.15]));s=f32(f32(f32(v[0]*v[0])+f32(v[1]*v[1]))+f32(v[2]*v[2]));norm=f32(1/f32(math.sqrt(s)));axis=[f32(x*norm) for x in v]
points=[]
for i in range(N):
 if i<7:p=[(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)][i]
 else:
  p=[((i*mul+shift)%8191-4095)/4096 for mul,shift in [(17,31),(53,11),(97,7)]];length=math.sqrt(sum(x*x for x in p));p=[x/length for x in p]
 points.extend(map(f32,p))
(B/'points_f32.bin').write_bytes(struct.pack('<'+'f'*len(points),*points))
cases=[];oracle=[]
for width in [3,4]:
 for ai,a in enumerate([axis,[0.,0.,1.]]):
  an=math.sqrt(sum(x*x for x in a));u=[x/an for x in a]
  for angle in [0.,.5,f32(math.pi/2)]:
   for repeat in [1,17]:
    # Independent quaternion vector rotation, negative source convention.
    theta=-repeat*angle;q0=math.cos(theta/2);q=[x*math.sin(theta/2) for x in u]
    offset=len(oracle)
    for i in range(N):
     p=points[3*i:3*i+3];cross=[q[1]*p[2]-q[2]*p[1],q[2]*p[0]-q[0]*p[2],q[0]*p[1]-q[1]*p[0]];t=[2*x for x in cross];cross2=[q[1]*t[2]-q[2]*t[1],q[2]*t[0]-q[0]*t[2],q[0]*t[1]-q[1]*t[0]];oracle.extend(p[k]+q0*t[k]+cross2[k] for k in range(3))
    cases.append(dict(n=N,width=width,axis_id=ai,axis=a,angle=angle,repeat=repeat,oracle_offset=offset,absolute_budget=5e-6+8e-7*repeat))
(B/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(oracle),*oracle))
(B/'fixture_description.json').write_text(json.dumps(dict(cases=cases,guard=12345.25,guard_words=12,source_axis=axis,source_axis_norm_squared=sum(x*x for x in axis),independent_oracle='Float64 normalized-axis quaternion by negative repeat*Float32angle, not replicated matrix coefficients',scope='Signed unit vectors, zero and six signed basis witnesses; finite declared source-rounded axis and exact z axis. Not arbitrary nonunit axes, original random-seed identity, or all-angle certification.',budget='Predeclared finite-case absolute 5e-6+8e-7*R; allows Float32 axis norm and repeated arithmetic/SFU rounding. Not formal all-input error bound.',invariants='Full xyz versus quaternion plus norm/axis-projection <=3*pointwisebudget; float4 fourth field exact positive zero; input/parameter/guards exact',timing='Sum24 separate host launch/contextcompletion windows, each divided by R; in-place across repeats, fresh buffer per case outside ROI; JIT/readback/reset outside. Five trials, average2-5.'),indent=2)+'\n')
