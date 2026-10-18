"""Reproduce frozen finite bit-copy study inputs; --write explicitly replaces fixture."""
from pathlib import Path
import struct,hashlib,sys,json
p=Path(__file__).resolve().parent;words=[]
for m,n,padding in [(256,127,1),(256,503,9)]:
 a=[float(((i*73+19)%1021)-510)/32 for i in range(m*n)]+[0.]*(m*padding)
 a[:6]=[0.,-0.,5e-324,-5e-324,123.5,-92.25]
 words.extend(struct.unpack('<Q',struct.pack('<d',x))[0] for x in a)
for m,n,padding in [(256,127,1),(256,503,9)]:
 for alpha in [0.,.1,.25,.5,.75,1.]:words.extend([m,n,padding,struct.unpack('<I',struct.pack('<f',alpha))[0],256,8,4,32,m*(n+padding)//8192])
b=struct.pack('<'+'Q'*len(words),*words)
if '--write' in sys.argv:(p/'fixture.bin').write_bytes(b)
assert (p/'fixture.bin').read_bytes()==b
print(json.dumps({'words':len(words),'sha256':hashlib.sha256(b).hexdigest(),'reproduced_exact':True}))
