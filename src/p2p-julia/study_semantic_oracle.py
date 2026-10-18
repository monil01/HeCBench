"""Independent exact full-byte oracle for named P2P study; no GPU operations."""
import struct,hashlib,json,sys
from pathlib import Path
CASES=(256,65536,16777216)
def chunks(n,scale=1):
 block=struct.pack('<4096f',*(float(i*scale) for i in range(4096)))
 for _ in range(n//4096):yield block
 if n%4096:yield block[:4*(n%4096)]
def expected_sha():
 h=hashlib.sha256();h.update(struct.pack('<f',3.))
 for n in CASES:
  h.update(struct.pack('<2f',float(n),10.))
  for data in chunks(n):h.update(data)
 for n in CASES:
  for scale in (1,1,4,2):
   h.update(struct.pack('<f',-12345.))
   for data in chunks(n,scale):h.update(data)
   h.update(struct.pack('<f',-12345.))
 return h.hexdigest()
if __name__=='__main__':
 expected=expected_sha();report={'expected_sha256':expected,'artifacts':[]}
 for name in sys.argv[1:]:
  h=hashlib.sha256()
  with open(name,'rb')as f:
   for block in iter(lambda:f.read(1<<20),b''):h.update(block)
  got=h.hexdigest();report['artifacts'].append({'path':name,'sha256':got,'matches_independent_full_byte_oracle':got==expected})
 print(json.dumps(report,indent=2));assert all(x['matches_independent_full_byte_oracle'] for x in report['artifacts'])
