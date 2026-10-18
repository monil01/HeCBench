from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent
# Codes BF16=1,F16=2,F32=3,I32=4,I8=5,U8=6; Python standard IEEE bit encodings.
pairs=[(s,t)for s,ts in [(1,[2,3,4,5,6]),(2,[2,3,4,5,6]),(3,[3,2,4,5,6]),(4,[4,3,2,5,6]),(5,[4,3,2,5,6]),(6,[4,3,2,5,6])]for t in ts]
def encode(x,c):
 if c==1:
  b=struct.unpack('<I',struct.pack('<f',x))[0];return (b+0x7fff+((b>>16)&1))>>16
 if c==2:return struct.unpack('<H',struct.pack('<e',x))[0]
 if c==3:return struct.unpack('<I',struct.pack('<f',x))[0]
 return int(x)% (1<<(32 if c==4 else 8))
def decode(b,c):
 if c==1:return struct.unpack('<f',struct.pack('<I',b<<16))[0]
 if c==2:return struct.unpack('<e',struct.pack('<H',b))[0]
 if c==3:return struct.unpack('<f',struct.pack('<I',b))[0]
 if c==4:return b-(1<<32)if b>=(1<<31)else b
 if c==5:return b-256 if b>=128 else b
 return b
fixture=[];oracle=[];records=[]
for s,t in pairs:
 for n in [17,257]:
  vals=([0.,-0.,.5,1.5,2.5,3.75,15.9375,31.5,63.875,126.5,127.,1+2**-11,1+3*2**-11,2**-24,2**-14,1+2**-8,1+3*2**-8]if s<=3 else([-32768,-1025,-257,-129,-128,-1,0,1,127,128,255,256,1023,1025,2049,32767]if s==4 else([-128,-127,-1,0,1,63,126,127]if s==5 else[0,1,127,128,129,254,255])))
  bits=[encode(vals[i%len(vals)],s)for i in range(n)];expected=[encode(decode(b,s),t)for b in bits];fixture += [n,s,t]+bits;oracle +=expected;records.append({'n':n,'source_code':s,'destination_code':t})
for name,v in [('fixtures_f64.bin',fixture),('oracle_f64.bin',oracle)]: (p/name).write_bytes(struct.pack('<'+'d'*len(v),*v))
(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent Python IEEE binary packing: BF16 round-to-nearest-even via high-bit integer rounding, IEEE half via struct e, float32 via struct f; integer truncation/modulo fixed widths, no GPU cast reused','cases':records,'input_values':'Finite floating values in0..127 valid for all float-to-integer targets, signed zero/halfway/subnormal values; fullsigned8/unsigned8 extrema and signed32 negatives/narrowing wrap','scope':'Float->integer out-of-range/NaN undefined or implementation-defined cases excluded; char source asserts signed8 on this host. Signed integer narrowing measured modulo2^width implementation policy.'},indent=2)+'\n')
