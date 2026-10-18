from pathlib import Path
import struct,math,json
B=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
eps=f32(1e-5);data=[];oracle=[];cases=[]
for N,H in [(7,1024),(5,1027)]:
 for fixture in [0,1]:
  offset=len(data);oo=len(oracle);x=[]
  for row in range(N):
   for col in range(H):
    if row==0:value=0.
    elif row==1:value=.5
    elif row==2:value=1. if col==H-1 else 0.
    elif row==3:value=(1e-4 if col%2 else -1e-4)
    else:value=((row*97+col*53+fixture*19)%2047-1023)/1024
    x.append(f32(value))
  gamma=[1. if fixture==1 else f32(((j*31)%1023-511)/256) for j in range(H)]
  data.extend(x+gamma)
  for row in range(N):
   xx=x[row*H:(row+1)*H];inv=1/math.sqrt(math.fsum(v*v for v in xx)/H+eps);oracle.extend(v*inv*g for v,g in zip(xx,gamma))
  cases.append(dict(n=N,h=H,fixture=fixture,input_offset=offset,oracle_offset=oo,unroll=4 if H%4==0 else 1))
(B/'fixtures_f32.bin').write_bytes(struct.pack('<'+'f'*len(data),*data));(B/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(oracle),*oracle))
paths=[dict(c,block_size=bt) for c in cases for bt in [32,64,128,256,512,1024]]
prefix=sum(2*(p['n']*p['h']+32)+p['h']+32 for p in paths);total=prefix+sum(p['n']*p['h']+32 for p in paths)
(B/'fixture_description.json').write_text(json.dumps(dict(cases=cases,paths=paths,guard=12345.25,guard_words=16,epsilon=eps,internal_repeats=20,input_prefix_count=prefix,complete_capture_count=total,independent_oracle='Float64 fsum squares /H +actualFloat32epsilon then sqrt andscale; independent of sourcewarp reduction ordering',budget='Predeclared pointwise 1e-5+2e-6*abs(oracle); finite bounded cases, source rsqrtf/FMA/tree rounding allowed; exactguards/input separately',fixtures='Allzero/constant/spike/epsilon-dominated alternating rows +signed varied rows, signed variedgamma orunitgamma; H1024 vector4/H1027scalar, sixoriginalblock sizes.',scope='Float32 sourceCLItype, fulloutput allrows/columns; no FP16/BF16type claim, no allinput theorem, no bandwidth from source access count.',timing='Sum24 independent sourcekernel hostlaunch/contextcompletion windows dividedbyR20; identicalrepeated overwrites, allsetup/copy/JIT/capture outside; fivecheckedtrials mean2-5.'),indent=2)+'\n')
