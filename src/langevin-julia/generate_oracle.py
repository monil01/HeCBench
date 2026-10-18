
from pathlib import Path
import struct,json,hashlib,sys,math,ctypes,decimal
P=Path(__file__).resolve().parent;N=4096
libpath=Path('/lib/x86_64-linux-gnu/libm.so.6').resolve();lib=ctypes.CDLL(str(libpath))
for n in ['coshf','sinhf','tanhf']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_float];f.restype=ctypes.c_float
lib.fmaf.argtypes=[ctypes.c_float]*3;lib.fmaf.restype=ctypes.c_float
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def polynomial(x):
 s=f32(x*x);r=f32(7.70960469e-8)
 for c in[-1.65101926e-6,2.03457112e-5,-2.10521728e-4,2.11580913e-3,-2.22220998e-2,8.33333284e-2]:r=lib.fmaf(r,s,f32(c))
 return lib.fmaf(r,x,f32(.25*x))
def exact(x):
 with decimal.localcontext()as ctx:
  ctx.prec=60;d=decimal.Decimal(x);e=(2*d).exp();return float((e+1)/(e-1)-1/d)
def main():
 files={};cases=[]
 for case in(1,2):
  step=f32(f32(1.79999)/N)
  vals=[f32(f32(-1.8)+f32(f32(i)*step))for i in range(N)]if case==1 else[(-1 if i%2 else 1)*((i*13%14+1)/8)for i in range(N)]
  policies=[f32(f32(lib.coshf(x)/lib.sinhf(x))-f32(1/x))for x in vals]+[f32(f32(1/lib.tanhf(x))-f32(1/x))for x in vals]+[polynomial(x)for x in vals]
  ideal=[exact(x)for x in vals];series=[x*(1/3-x*x/45+2*x**4/945-x**6/4725)for x in vals]
  for name,b in[(f'study_case{case}_f32.bin',struct.pack('<'+'f'*N,*vals)),(f'oracle_case{case}_f64.bin',struct.pack('<'+'d'*(5*N),*(policies+ideal+series)))]:
   (P/name).write_bytes(b);files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  cases.append(dict(case=case,min=min(vals),max=max(vals),min_abs=min(map(abs,vals)),source_float32_initializer=case==1,zero_included=False))
 (P/'fixture_provenance.json').write_text(json.dumps(dict(n=N,cases=cases,files=files,policy_oracle='CPUlibmFloat32coshf/sinhf/tanhf +roundedFloat32div/subtract;fmaf polynomial exactsource coefficients/order',mathematical_oracle='60decimaldigits exp coth(x)-1/x; source truncatedx7series separately retained',libm=dict(path=str(libpath),sha256=hashlib.sha256(libpath.read_bytes()).hexdigest()),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version),indent=2)+'\n')
if __name__=='__main__':main()
