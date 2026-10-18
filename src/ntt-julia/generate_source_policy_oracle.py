from pathlib import Path
import ctypes,struct,json,shutil,hashlib
R=Path('/home/imo/HeCBench_julia');B=R/'src/ntt-julia';N=65536
import subprocess,tempfile
ntt_tmp=tempfile.TemporaryDirectory(prefix='ntt-policy-cpu-')
ntt_so=Path(ntt_tmp.name)/'policy.so'
subprocess.run(['g++','-O3','-shared','-fPIC',str(B/'native_modular_cpu_witness.cpp'),'-o',str(ntt_so)],check=True)
lib=ctypes.CDLL(str(ntt_so));a=lib.ntt8_original;a.argtypes=[ctypes.POINTER(ctypes.c_ulong)];s=lib.shift_original;s.argtypes=[ctypes.c_ulong,ctypes.c_int];s.restype=ctypes.c_ulong;mul=lib.multiply_original;mul.argtypes=[ctypes.c_ulong,ctypes.c_ulong];mul.restype=ctypes.c_ulong
P=0xffffffff00000001;MASK=(1<<64)-1
shutil.copy2(B/'oracle_u32.bin',B/'ideal_field_oracle_u32.bin');data=list(struct.unpack('<'+'Q'*(3*N),(B/'inputs_u64.bin').read_bytes()));out=[]
def add(x,y):
 r=(x+y)&MASK
 if r<x:r=(r+0xffffffff)&MASK
 if r>=P:r-=P
 return r
def sub(x,y):
 r=(x-y)&MASK
 if r>x:r=(r-0xffffffff)&MASK
 return r
for c in range(3):
 z=[None]*N
 for b in range(128):
  sh=[0]*512
  for t in range(64):
   f=(b<<9)|((t&0x3e)<<3)|(t&1);v=(ctypes.c_ulong*8)(*[data[c*N+(f|(i<<1))]for i in range(8)]);a(v)
   for i in range(8):sh[(t<<3)|i]=s(v[i],((t&1)<<2)*i*3)
  for t in range(64):
   f=((t&0x38)<<3)|(t&7);tm=(b<<9)|f
   for i in range(8):
    j=i&~1;x,y=sh[f|(j<<3)],sh[f|((j+1)<<3)];v=add(x,y)if i%2==0 else sub(x,y);idx=(((tm|(i<<3))&15)<<12)|((tm|(i<<3))>>4);z[idx]=mul(v,18446462594437939201)&0xffffffff
 assert None not in z;out.extend(z)
(B/'source_policy_oracle_u32.bin').write_bytes(struct.pack('<'+'I'*len(out),*out));print('Generated full original-source policy oracle; compare with independent ideal_field_oracle_u32.bin separately.')
