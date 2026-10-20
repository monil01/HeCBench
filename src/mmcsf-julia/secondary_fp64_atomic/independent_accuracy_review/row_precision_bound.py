from pathlib import Path
import json,struct,array,ctypes,math,hashlib
from fractions import Fraction
P=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');d=json.load(open(R/'src/mmcsf-julia/float32_accumulation_control.json'));path=R/d['canonical'];wanted={0:'f',6:'Q',7:'Q'};a={}
with open(path,'rb')as f:
 magic,n=struct.unpack('<QQ',f.read(16));assert magic==0x4d4d435346494e50 and n==10
 for k in range(n):
  size,kind=struct.unpack('<QQ',f.read(16))
  if k in wanted:
   v=array.array(wanted[k]);v.frombytes(f.read(size));a[k]=v
  else:f.seek(size,1)
lib=ctypes.CDLL('libm.so.6');fma=lib.fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def units(v):
 n,q=float(v).as_integer_ratio();assert q&(q-1)==0;return n<<(149-(q.bit_length()-1))
coo=0;sumc=0;absc=0;m=0;nz=0
for fiber in range(len(a[7])-1):
 if a[6][fiber]!=d['row']:continue
 tmp=0.
 for z in range(a[7][fiber],a[7][fiber+1]):
  tmp=fma(a[0][z],1.5,tmp);coo+=units(a[0][z]);nz+=1
 c=f32(f32(tmp*2.5)*3.5);uc=units(c);sumc+=uc;absc+=abs(uc);m+=1
assert nz==d['input_nonzeros']and m==d['rounded_source_fiber_contributions'];exact=Fraction(coo*105,8*(1<<149));private=Fraction(sumc,1<<149);sumabs=Fraction(absc,1<<149);privateerror=abs(private-exact);gamma=Fraction(m-1,(1<<53)-(m-1));accbound=gamma*sumabs;projbound=Fraction(1,1<<24)*(abs(private)+accbound)+Fraction(1,1<<150);budget=Fraction(1,100)+Fraction(2,100000)*abs(exact)
report=dict(status='bounded_CPU_rounding_model_diagnostic_only',row=d['row'],mode=0,contributions=m,nonzeros=nz,exact_COO_sum=float(exact),rounded_F32_contributions_exact_sum=float(private),sum_absolute_F32_contributions=float(sumabs),private_rounding_abs_error=float(privateerror),FP64_atomic_any_order_gamma_bound=float(accbound),FP32_projection_round_to_nearest_bound=float(projbound),total_FP64_bound=float(privateerror+accbound),total_projected_FP32_bound=float(privateerror+accbound+projbound),old_strict_budget=float(budget),model_bounds_below_old_budget=privateerror+accbound+projbound<budget,scope='ExactrationalerrorarithmeticforCPUmodeledsourceF32fibercontributionmultiset. libmfmaf suppliesF32fusedmultiplyadd. ActualGPUatomicorderunknown; actualGPUprivatecontributionsnotcaptured, so notproofGPUvariantpasses or isolationofexactnativefailure. AppliesONLYrow762mode0andhypothesizedsourceFFMA/roundingpolicy, notall7branches/rows.',assumptions=['FiniteF32tensor/factors,sourceleftassociatedF32products unchanged','SourceprivateFFMA graphmatchesCPUmodeledfmaf','PromotedatomicaddroundsnearestinF64withoutoverflow; everycontributionexactlyconvertedF32toF64','ProjectionroundstonearestF32withoutoverflow; originalstrictbudgetunchanged'],source={'canonical':str(path),'declared_canonical_sha256':d['canonical_sha256']},GPU_execution=False)
(P/'row_precision_bound.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
