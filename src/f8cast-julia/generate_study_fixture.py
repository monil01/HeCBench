import struct,math,bisect,json
from pathlib import Path
p=Path(__file__).resolve().parent
tobits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
tof=lambda b:struct.unpack('<f',struct.pack('<I',b))[0]
def values(m,bias,last):
 return [math.ldexp(float(c&((1<<m)-1)),1-bias-m) if c>>m==0 else math.ldexp(1+(c&((1<<m)-1))/(1<<m),(c>>m)-bias) for c in range(last+1)]
tables=[values(3,7,127),values(2,15,124)]
bits=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fffffff,0xffffffff]
for tab in tables:
 for x in tab:
  b=tobits(x)
  for s in [0,0x80000000]:bits.append(b|s)
 for x,y in zip(tab,tab[1:]):
  b=tobits((x+y)/2)
  for delta in [-1,0,1]:
   for s in [0,0x80000000]:bits.append((b+delta)|s)
state=19937
for _ in range(3073):
 state^=(state<<13)&0xffffffff;state^=state>>17;state^=(state<<5)&0xffffffff;state&=0xffffffff;bits.append(state)
if len(bits)%256==0:bits.append(tobits(449.0))
def encode(b,fmt,sat):
 sign=(b>>24)&128;mag=b&0x7fffffff;tab=tables[fmt];last=len(tab)-1
 if mag>0x7f800000:return 127 if sat else 127|sign
 if mag==0x7f800000:return ((last-1) if sat else last)|sign
 x=abs(tof(b));k=bisect.bisect_left(tab,x);choices={min(k,last),max(0,min(k-1,last))};best=min(choices,key=lambda c:(abs(tab[c]-x),c&1))
 if sat:best=min(best,last-1)
 return best|sign
n=len(bits);expected=bytearray()
for fmt,sat in [(0,False),(0,True),(1,False),(1,True)]:
 expected.extend(encode(b,fmt,sat) for b in bits);expected.extend([165]*31)
(p/'fixture_bits.bin').write_bytes(struct.pack('<'+'I'*n,*bits));(p/'expected_codes.bin').write_bytes(expected)
(p/'fixture_definition.json').write_text(json.dumps({'elements':n,'output_padding_per_path':31,'path_order':['e4m3fn_reference_nonsaturating','e4m3_vendor_satfinite','e5m2_reference_nonsaturating','e5m2_vendor_satfinite'],'generation':'Both signs of all positive encoding codepoints and Float32 below/tie/above every adjacent encoding midpoint, special rawbit NaNs/Inf/zero/subnormal plus3073 xorshift32bitpatterns seed19937. UInt32serializedprefix preservesallNaNpayloads.','independent_oracle':'Enumerate exact binary format values as Float64; nearest-even codebook distance, including virtual480(E4NaN) and65536(E5Inf) overflow endpoints. SATFINITE capsfinitecode; canonicalpositiveNaN; reference retainsNaNsign.','software_fp8_source':'OriginalCUDA kernels.h bit manipulations and Float32 denorm magic; original vendor conversion onsm70 usesCUDA12.0header Float64softwarefallback, native ISA gatedsm90.'},indent=2)+'\n')
print(n)
