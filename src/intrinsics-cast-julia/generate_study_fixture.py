#!/usr/bin/env python3
"""Independent IEEE rounding/bit-semantics oracle; check or create named fixture."""
import math,struct,hashlib,json,argparse
from pathlib import Path
N=65536;MASK32=2**32-1;MASK64=2**64-1;ROOT=Path(__file__).resolve().parent
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits32(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def bits64(x):return struct.unpack('<Q',struct.pack('<d',x))[0]
def frombits32(x):return struct.unpack('<f',struct.pack('<I',x&MASK32))[0]
def frombits64(x):return struct.unpack('<d',struct.pack('<Q',x&MASK64))[0]
def round4(x):return [math.floor(x),round(x),math.ceil(x),math.trunc(x)]
def xor4(x):
 y=0
 for v in round4(x):y^=v
 return y
def fpword(x):
 u=bits64(x);v=bits32(f32(x));a=xor4(x)^xor4(f32(x));r1=((u>>32)^(u&MASK32)^a^v)&MASK32;r2=(a^v)&MASK32;r3=(a^u)&MASK64;r4=a&MASK64
 return (((r1+r2)&MASK32)+((r3+r4)&MASK64))&MASK64
def next32(x,up):
 if x==0:return frombits32(1 if up else 0x80000001)
 u=bits32(x);return frombits32(u+(1 if (x>0)==up else -1))
def cvt_int(x,width,mode):
 y=f32(x)if width==32 else float(x)
 if mode=='rd'and y>x:y=next32(y,False)if width==32 else math.nextafter(y,-math.inf)
 if mode=='ru'and y<x:y=next32(y,True)if width==32 else math.nextafter(y,math.inf)
 if mode=='rz'and ((x>0 and y>x)or(x<0 and y<x)):y=next32(y,x<0)if width==32 else math.nextafter(y,0.)
 return y
def intword(x):
 low=x&MASK32;signedlow=low if low<2**31 else low-2**32;r1=f32(frombits64(x))
 for val in [signedlow,low]:
  for mode in ['rd','rn','ru','rz']:r1=f32(r1+cvt_int(val,32,mode))
 r1=f32(r1+frombits32(low));r1=f32(r1+frombits32(low))
 for val in [x,x&MASK64]:
  for mode in ['rd','rn','ru','rz']:r1=f32(r1+cvt_int(val,32,mode))
 r2=float(signedlow);r2+=float(low)
 for val in [x,x&MASK64]:
  for mode in ['rd','rn','ru','rz']:r2+=cvt_int(val,64,mode)
 r2+=frombits64(x)
 return bits64(float(r1)+r2)
def payloads():
 xs=[22.44/(i+1)for i in range(N)];word=0x403670A3D70A3D71
 return {'input_from_fp_f64.bin':struct.pack('<'+'d'*N,*xs),'input_to_fp_i64.bin':struct.pack('<'+'Q'*N,*([word]*N)),'expected_from_fp_i64.bin':struct.pack('<'+'Q'*N,*map(fpword,xs)),'expected_to_fp_i64.bin':struct.pack('<Q',intword(word))*N}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--create',action='store_true');args=ap.parse_args();rows={}
 for name,data in payloads().items():
  path=ROOT/name
  if args.create:
   if path.exists():assert path.read_bytes()==data
   else:path.write_bytes(data)
  else:assert path.read_bytes()==data
  rows[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 print(json.dumps({'n':N,'oracle':'Pythonintegerexactbits, IEEEbinary32via struct, directedint->float comparesexactPythonint andstepsneighbor; defaultRN ties-to-even','source_initializer_replaced_by_defined_fixture':True,'files':rows},indent=2))
