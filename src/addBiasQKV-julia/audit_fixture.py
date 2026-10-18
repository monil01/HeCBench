"""Independent array oracle: enumerate finite E4M3 grid, nearest-even code selection.
Does not reuse C++ reference conversion or Julia encoder implementation.
"""
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent
codes=np.arange(127,dtype=np.uint8);exp=codes>>3;mant=codes&7
levels=np.where(exp==0,mant.astype(np.float64)*2**-9,np.ldexp((8+mant).astype(np.float64),exp.astype(int)-10)).astype(np.float32)
decode=np.r_[levels,np.nan,-levels,np.nan].astype(np.float32)
def bf(x):
 x=np.array(x,dtype=np.float32);u=x.view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def encode(x):
 a=np.abs(x);hi=np.searchsorted(levels,a,side='left').clip(0,126);lo=(hi-1).clip(0,126);dl=a-levels[lo];dh=levels[hi]-a;k=np.where((dl<dh)|((dl==dh)&((lo&1)==0)),lo,hi);k=np.where(a>=448,126,k);return (k.astype(np.uint8)|(np.signbit(x).astype(np.uint8)<<7))
rows=[]
for tag,m,h,d in [('original',8192,12,64),('axis',5,3,16)]:
 s=np.fromfile(P/f'study_{tag}_src_u8.bin','u1').reshape(m,3,h,d);b=np.fromfile(P/f'study_{tag}_bias_u16.bin','<u2').astype(np.uint32).__lshift__(16).view(np.float32).reshape(1,3,h,d);sc=np.fromfile(P/f'study_{tag}_scales_f32.bin','<f4');expected=np.fromfile(P/f'study_{tag}_expected_u8.bin','u1').reshape(m,h,3,d);assert np.all(np.isfinite(decode[s]));bad=0;nonrounded=0;maxquant=0.
 for start in range(0,m,256):
  a=decode[s[start:start+256]];v=bf(bf(bf(a)*bf(sc[0]))+b);v=bf(v*bf(sc[1]));out=encode(v).transpose(0,2,1,3);goal=expected[start:start+256];bad+=np.count_nonzero(out!=goal);direct=encode((a*sc[0]+b)*sc[1]).transpose(0,2,1,3);nonrounded+=np.count_nonzero(direct!=goal);maxquant=max(maxquant,float(np.max(abs(decode[out]-v.transpose(0,2,1,3)))))
 assert bad==0,(tag,bad)
 rows.append({'case':tag,'all_sourceCPU_oracle_bytes_equal_independent_grid_oracle':True,'checked_bytes':int(s.size),'without_intermediate_BF16rounding_different_FP8bytes':int(nonrounded),'maximum_final_FP8_quantization_absolute_error_vs_rounded_BF16_value':maxquant,'scales_raw':sc.tolist(),'scales_BF16rounded':bf(sc).tolist(),'all_finite_source_FP8codes':True,'source_axis_shape':[m,3,h,d],'output_axis_shape':[m,h,3,d]})
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in P.glob('study_*.bin')};report={'numpy':np.__version__,'oracle_method':'Independent enumerated positivefiniteE4M3grid/nearestcode/ties-even-sign and vectorizedRNEBF16at eachoperation; sourceCPUoracle usedonlyascomparison. Allfiniteinputs, overflow saturation implementation present but not exercised bythese scales, NaN/Inf outofscope.','cases':rows,'files':files,'generator':{'path':'generate_study_fixture.cpp','sha256':hashlib.sha256((P/'generate_study_fixture.cpp').read_bytes()).hexdigest()},'source_oracle':{'path':'../addBiasQKV-cuda/reference.h','sha256':hashlib.sha256((P.parent/'addBiasQKV-cuda/reference.h').read_bytes()).hexdigest()}}
(P/'fixture_provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(rows)
