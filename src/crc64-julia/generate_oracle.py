"""CRC oracle uses independent bit-by-bit GF(2), not slicing/interleaving tables."""
from pathlib import Path
import re,hashlib,json,struct
p=Path(__file__).resolve().parent;s=(p.parent/'crc64-cuda/CRC64.cu').read_text();poly=0xc96c5795d7870f42;mask=(1<<64)-1
files={}
def save(name,raw):
 (p/name).write_bytes(raw);files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def crc(data):
 cs=mask
 for byte in data:
  cs ^=byte
  for _ in range(8):cs=(cs>>1)^(poly if cs&1 else 0)
 return cs^mask
for name,size in [('crc64_table',1024),('crc64_interleaved_table',1024),('crc64_x_pow_2n',64)]:
 body=s[s.index('static const uint64_t '+name):];body=body[:body.index('};')];v=[int(x,16) for x in re.findall(r'UINT64_C\(0x([0-9a-f]+)\)',body)];assert len(v)==size;save(name+'.bin',struct.pack('<'+'Q'*size,*v))
out=[]
for case,n in enumerate([65555,65677],1):
 data=bytes(((i*73+(i>>3)*19+(i*i>>7)+case*47)&255) for i in range(n));save(f'study_input{case}_u8.bin',data);bpt=n//64;sizes=[bpt]*63+[n-63*bpt];parts=[crc(data[i*bpt:i*bpt+sz]) for i,sz in enumerate(sizes)];full=crc(data);check=crc(data+struct.pack('<Q',full^mask));assert check==mask;out.append(dict(case=case,n=n,threads=64,bpt=bpt,sizes=sizes,partial_crc=parts,full_crc=full,appended_check_crc=check,split=12345))
known=crc(b'123456789');assert known==0x995dc9bbdf1939fa
flat=[]
for c in out:flat+=c['sizes']+c['partial_crc']+[c['full_crc']]*10+[c['appended_check_crc'],c['full_crc']]
flat+=[known];assert len(flat)==281;save('oracle_u64.bin',struct.pack('<281Q',*flat))
meta=dict(polynomial_reversed=hex(poly),initial_xor=hex(mask),final_xor=hex(mask),oracle='independentbit-by-bitreflectedECMA182/CRC64XZ',known_message='123456789',known_crc=known,cases=out,files=files,native_source_sha256=hashlib.sha256(s.encode()).hexdigest(),native_thread_coverage_scope='Eachcasefloor(nbytes/1024)=64,grid1block64coversallworkers; arbitrarynonmultiple64nthreadsnotcertified',fixture_generator='Pythonstdlibonly')
(p/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:len(v) if isinstance(v,list) else v for k,v in meta.items() if k in ['cases','known_crc']})
