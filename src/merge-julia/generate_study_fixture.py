#!/usr/bin/env python3
from pathlib import Path
import struct,json,hashlib
P=Path(__file__).resolve().parent;N=100000;PAD=1024;meta={'N':N,'padding':PAD,'blocks':112,'threads':128,'generator':'generate_study_fixture.py','types':{}}
for mode,fmt,maximum,base in [('uint32','I',2**32-1,0),('float32','f',struct.unpack('<f',bytes.fromhex('ffff7f7f'))[0],0),('uint64','Q',2**64-1,0x9000000000000000),('float64','d',float.fromhex('0x1.fffffffffffffp+1023'),0)]:
 a=[base+1+(i//3)*5 for i in range(N)];b=[base+1+(i//4)*7 for i in range(N)];merged=[];take_a=take_b=0;paths_a=[];paths_b=[];diags=[g*(2*N)//112 for g in range(113)];di=0
 while take_a+take_b<2*N:
  rank=take_a+take_b
  while di<113 and diags[di]==rank:paths_a.append(take_a);paths_b.append(take_b);di+=1
  if take_b==N or(take_a<N and a[take_a]<=b[take_b]):merged.append(a[take_a]);take_a+=1
  else:merged.append(b[take_b]);take_b+=1
 paths_a.append(N);paths_b.append(N);assert len(paths_a)==113
 raw=struct.pack('<'+str(2*(N+PAD))+fmt,*(a+[maximum]*PAD+b+[maximum]*PAD));out=struct.pack('<'+str(2*N)+fmt,*merged);path=struct.pack('<226I',*(paths_a+paths_b))
 for stem,data in [('input',raw),('expected',out),('paths',path)]: (P/f'study_{stem}_{mode}.bin').write_bytes(data)
 meta['types'][mode]={'source_domain':'Positive finite/unsigned values strictly within source sentinels; duplicates/ties; UInt64 high-bit fixture extends RNG range, exact UInt64 storage/comparison.','input_prefix_count':2*(N+PAD),'expected_output_values':2*N,'path_values':226,'files':{q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in P.glob(f'study_*_{mode}.bin')}}
(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print({m:v['files']for m,v in meta['types'].items()})
