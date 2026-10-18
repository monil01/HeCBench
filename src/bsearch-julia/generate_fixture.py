from pathlib import Path
import struct,math,json,hashlib
sizes=[2,3,8,9,17,32,33,257,1025];words=[];cases=[]
def nxt(x,direction,bits):
 if bits==64:return math.nextafter(x,math.inf if direction>0 else -math.inf)
 if x==0:return struct.unpack('<f',struct.pack('<I',1 if direction>0 else 0x80000001))[0]
 b=struct.unpack('<I',struct.pack('<f',x))[0];b+=1 if (x>0)==(direction>0) else -1;return struct.unpack('<f',struct.pack('<I',b))[0]
def word(x,bits):return struct.unpack('<I' if bits==32 else '<Q',struct.pack('<f' if bits==32 else '<d',x))[0]
for bits in [32,64]:
 for size in sizes:
  a=[-8.0]
  for i in range(1,size):a.append(a[-1]+(1+(i%5))/8)
  n=size-1;z=[]
  for j in range(512):
   k=(j*73+19)%n;mode=j%4
   z.append(a[k] if mode==0 else nxt(a[k],1,bits) if mode==1 else nxt(a[k+1],-1,bits) if mode==2 else (a[k]+a[k+1])/2)
  z[0]=a[0];z[1]=nxt(a[-1],-1,bits);z[2]=a[n//2]
  assert all(a[0]<=x<a[-1] and math.isfinite(x) for x in z)
  offset=len(words);words.extend(word(x,bits) for x in a+z);cases.append({'dtype':'Float'+str(bits),'a_size':size,'n':n,'queries':512,'word_offset':offset})
raw=struct.pack('<%dQ'%len(words),*words);path=Path(__file__).with_name('fixture.bin')
if path.exists():
 if path.read_bytes()!=raw:raise RuntimeError('Existingfixturemismatch; preservebeforeinvestigating')
else:path.write_bytes(raw)
print(json.dumps({'cases':cases,'prefix_count':len(words),'fixture_sha256':hashlib.sha256(raw).hexdigest()}))
