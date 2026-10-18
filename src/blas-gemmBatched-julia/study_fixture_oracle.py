from pathlib import Path
import struct, math, json
p=Path(__file__).resolve().parent
fixtures=[];oracle=[];bounds=[];cases=[];U=17
for code,fmt,unit in [(0,'e',2**-11),(1,'f',2**-24),(2,'d',2**-53)]:
 cast=lambda x:struct.unpack('<'+fmt,struct.pack('<'+fmt,x))[0]
 for mode in [0,1]:
  for batch in [3,17]:
   aa=[cast(((i*17+3)%19-9)/(16 if mode==0 else 13)) for i in range(U*U*batch)]
   bb=[cast(((i*11+7)%17-8)/(16 if mode==0 else 11)) for i in range(U*batch)]
   start=len(fixtures);fixtures.extend([code,mode,batch,U,len(aa),len(bb)]+aa+bb);cases.append({'code':code,'mode':mode,'batch':batch,'offset':start,'data_count':len(aa)+len(bb)})
   for m in range(2,U+1):
    for b in range(batch):
     for row in range(U):
      if row>=m:oracle.append(-1024.);bounds.append(0.);continue
      products=[aa[b*U*U+row+k*U]*bb[b*U+k] for k in range(m)];v=math.fsum(products);gamma=(2*m*unit)/(1-2*m*unit);oracle.append(v);bounds.append(gamma*math.fsum(abs(x)for x in products)+4*unit*abs(v))
for name,data in [('fixtures_f64.bin',fixtures),('oracle_f64.bin',oracle),('bounds_f64.bin',bounds)]:
 with(p/name).open('wb')as f:f.write(struct.pack('<'+'d'*len(data),*data))
(p/'fixture_layout.json').write_text(json.dumps({'upper':17,'sizes':list(range(2,18)),'types':['Float16','Float32','Float64'],'cases':cases,'prefix_count':len(fixtures),'payload_count':len(oracle),'oracle':'Independent Float64 math.fsum over exact stored scalar inputs, no cuBLAS/CPUsource checker reuse','bound':'gamma_(2m)(unit)*sum(abs(products))+4unit*abs(expected), unit=2^-11/-24/-53. Padding exact. Source storage/API precision bound, internal vendor accumulation not inferred.','source_alpha':1,'source_beta':0},indent=2)+'\n')
