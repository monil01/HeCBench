import struct,json,itertools,math
from pathlib import Path
p=Path(__file__).resolve().parent
cases=[];fixture=[];oracle=[]
G=-777.0
for ci,L in enumerate([4,6,4]):
 n=L**4;e=n//2
 coords=sorted(itertools.product(range(L),repeat=4),key=lambda c:(sum(c)%2, c[0]+L*(c[1]+L*(c[2]+L*c[3]))))
 lookup={c:i for i,c in enumerate(coords)}
 def nb(c,d,s):
  x=list(c);x[d]=(x[d]+s)%L;return lookup[tuple(x)]
 maps=[[nb(c,d,step) for c in coords for d in range(4)] for step in [1,-1,3,-3]]
 src=[float(((i*17+ci*3)%31)-15)/32 for i in range(n*6)]
 fat=[float(((i*7+ci)%29)-14)/64 for i in range(n*72)]
 lng=[float(((i*11+ci*5)%23)-11)/128 for i in range(n*72)]
 if ci==2:
  src=[v for i in range(n) for v in [0.25,-0.125,0.5,0.375,-0.25,0.125]]
  fat=[float(r==c and a==0) for i in range(n) for d in range(4) for r in range(3) for c in range(3) for a in range(2)]
  lng=[v/8 for v in fat]
 def mat(arr,i,d):return [[complex(arr[i*72+d*18+r*6+c*2],arr[i*72+d*18+r*6+c*2+1]) for c in range(3)] for r in range(3)]
 def vec(i):return [complex(src[i*6+c*2],src[i*6+c*2+1]) for c in range(3)]
 backs=[]
 for arr,mp in [(fat,maps[1]),(lng,maps[3])]:
  b=[G]*(n*72)
  for i in range(e):
   for d in range(4):
    m=mat(arr,mp[4*i+d],d)
    for r in range(3):
     for c in range(3):
      v=m[c][r].conjugate();b[i*72+d*18+r*6+c*2:r*0+i*72+d*18+r*6+c*2+2]=[v.real,v.imag]
  backs.append(b)
 dst=[G]*(n*6)
 for i in range(e):
  z=[0j]*3
  for arr,mp,sign in [(fat,maps[0],1),(lng,maps[2],1),(backs[0],maps[1],-1),(backs[1],maps[3],-1)]:
   for d in range(4):
    m=mat(arr,i,d);v=vec(mp[4*i+d])
    for r in range(3):z[r]+=sign*sum(m[r][c]*v[c] for c in range(3))
  for r,v in enumerate(z):dst[i*6+r*2:i*6+r*2+2]=[v.real,v.imag]
 if ci==2:assert all(x==0 for x in dst[:e*6])
 prefix=[float(L),float(n)]+src+fat+lng+sum(maps,[])
 fullinputs=[]
 for a,g in [(src,6),(fat,18),(lng,18)]:fullinputs+= [G]*g+a+[G]*g
 for a in maps:fullinputs += [999999.0]*4+a+[999999.0]*4
 output=[G]*6+dst+[G]*6
 for b in backs:output += [G]*18+b+[G]*18
 output+=fullinputs
 cases.append(dict(L=L,n=n,prefix_count=len(prefix),output_count=len(output),identity_constant_control=ci==2))
 fixture+=prefix;oracle+=output
p.joinpath('fixture.bin').write_bytes(struct.pack('<%sd'%len(fixture),*fixture))
p.joinpath('oracle.bin').write_bytes(struct.pack('<%sd'%len(oracle),*oracle))
p.joinpath('fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(fixture),output_count=len(oracle),atol=2e-5,rtol=2e-6,oracle='coordinate-indexed complex matrix-vector algebra with backward conjugate transpose; dyadic inputs exactly representable, constant identity-link null control'),indent=2))
