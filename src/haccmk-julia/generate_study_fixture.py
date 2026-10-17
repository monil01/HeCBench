#!/usr/bin/env python3
"""Canonical source-size HACC Float32 recurrence; default checks saved bytes."""
import struct,hashlib,argparse
from pathlib import Path

def f32(x):
 return struct.unpack('<f',struct.pack('<f',x))[0]

def fixture():
 n=15000;x=[0.]*n;y=[0.]*n;z=[0.]*n;m=[0.]*n;m[0]=2.
 dx=f32(1./n);dy=f32(2./n);dz=f32(3./n);coef=f32(.01)
 for i in range(1,n):
  x[i]=f32(x[i-1]+dx);y[i]=f32(y[i-1]+dy);z[i]=f32(z[i-1]+dz)
  m[i]=f32(f32(f32(i)*coef)+x[i])
 return struct.pack('<'+str(4*n)+'f',*(x+y+z+m))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args();data=fixture()
 if args.output:
  with args.output.open('xb')as f:f.write(data)
 elif data!=Path(__file__).with_name('study_input_f32.bin').read_bytes():raise SystemExit('saved bytes differ')
 print(hashlib.sha256(data).hexdigest())
