from pathlib import Path
import struct,ctypes,json,hashlib
N=2048;libc=ctypes.CDLL(None);libc.srand(123);canonical=[libc.rand()%256 for _ in range(N)];varied=[((i*37+11)%127)-63 for i in range(N)]
for j,i in enumerate([0,1,63,64,127,128,255,256,511,512,1023,1024,2046,2047]):varied[i]=(j%9)-4
inputs=[canonical,varied];types=['Float64','Float32','Int16'];words=[];cases=[];groups=[]
def enc(x,T):
 if T=='Int16':return struct.unpack('<H',struct.pack('<h',x))[0]
 return struct.unpack('<Q' if T=='Float64' else '<I',struct.pack('<d' if T=='Float64' else '<f',x))[0]
for fixture in range(2):
 for T in types:
  cases.append({'fixture':fixture,'dtype':T,'offset':len(words),'length':N});words.extend(enc(x,T) for x in inputs[fixture])
for fixture in range(2):
 for mw in [3,5,7,9]:
  mask=[1]*mw if fixture==0 else [-2,1,0,3,-1,2,-2,0,1][:mw]
  for ti,T in enumerate(types):
   groups.append({'fixture':fixture,'dtype':T,'input_case':fixture*3+ti,'mask_width':mw,'mask_offset':len(words),'block_sizes':[64,128,256,512,1024],'paths':15});words.extend(enc(x,T) for x in mask)
raw=struct.pack('<%dQ'%len(words),*words);f=Path(__file__).with_name('fixture.bin')
if f.exists():
 if f.read_bytes()!=raw:raise RuntimeError('Fixture mismatch; preserve existing source-hostlibc evidence before regenerating.')
else:f.write_bytes(raw)
print(json.dumps({'prefix_count':len(words),'sha256':hashlib.sha256(raw).hexdigest()}))
