"""Independent precision-aware dot-product hierarchy and exact input-bit fixtures."""
from pathlib import Path
import numpy as np,json,hashlib,sys
P=Path(__file__).resolve().parent;N=32768;GRIDS=[256,128,64,32,16]
def fma_acc(a,b,grid,T):
 count=grid*128;acc=np.zeros((count,2),dtype=T)
 for start in range(0,N,count):acc=(a[start:start+count].astype(float)*b[start:start+count].astype(float)+acc.astype(float)).astype(T)
 return acc.reshape(grid,128,2)
def tree_pair(x):
 x=x.copy()
 for off in (64,32,16,8,4,2,1):x[:,:off]=(x[:,:off].astype(float)+x[:,off:2*off].astype(float)).astype(x.dtype)
 return (x[:,0,0].astype('<f4')+x[:,0,1].astype('<f4')).astype('<f4')
def cub_block(x):
 z=(x[:,:,0]+x[:,:,1]).astype('<f4').reshape(-1,4,32)
 for off in (1,2,4,8,16):
  old=z.copy();z[:,:,:32-off]=(old[:,:,:32-off].astype(float)+old[:,:,off:].astype(float)).astype('<f4')
 result=z[:,0,0].copy()
 for warp in (1,2,3):result=(result.astype(float)+z[:,warp,0].astype(float)).astype('<f4')
 return result
files={};cases=[];records=[]
for case in (1,2):
 q=np.arange(N*2,dtype=np.int64)
 if case==1:
  a=(((q*13)%31-15)/16+1/256).astype('<f2').reshape(N,2);b=(((q*7)%29-14)/16-1/128).astype('<f2').reshape(N,2)
 else:
  a=np.full((N,2),np.float16(np.sqrt(32752./N)),dtype='<f2');b=a.copy()
 bits=np.concatenate([a.ravel().view('<u2'),b.ravel().view('<u2')]);f=P/f'study_case{case}_u16.bin';bits.tofile(f);files[f.name]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 ideal=float(np.sum(a.astype(float)*b.astype(float)));values=[]
 for grid in GRIDS:
  for mode in (1,2,3):
   if mode==1:block=tree_pair(fma_acc(a,b,grid,'<f2'))
   elif mode==2:block=tree_pair(fma_acc(a,b,grid,'<f4'))
   else:block=cub_block(fma_acc(a,b,grid,'<f4'))
   expected=float(np.sum(block.astype(float)));values.append(expected);records.append(dict(case=case,grid=grid,mode=mode,expected=expected,ideal_real_dot_of_quantized_inputs=ideal,atomic_sum_abs_blocks=float(np.sum(abs(block.astype(float)))),oracle='Float64 exact product-plus-half/float accumulator rounded after each source FMA and source tree/warp operation; final Float32 atomic-order variability admitted by declared budget.'))
 values.append(float(np.float16(ideal)));roundedprod=(a.astype(float)*b.astype(float)).astype('<f2');values.append(float(np.sum(roundedprod.astype(float))));f=P/f'oracle_case{case}_f64.bin';np.array(values,dtype='<f8').tofile(f);files[f.name]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest());cases.append(dict(case=case,ideal_real_dot_of_quantized_inputs=ideal,source_unquantized_analytical_claim=65504 if case==2 else None,input_half_scalar_values_per_vector=N*2,input_bit_values=len(bits),expected=values,cuBLAS_half_result_expected=values[-2],cub_rounded_half_product_sum=values[-1],finite_all=True))
(P/'fixture_provenance.json').write_text(json.dumps(dict(cases=cases,modes=records,files=files,half2_count=N,grids=GRIDS,atol=.02,rtol=2e-6,oracle='Independent NumPy Float64 products plus explicit RNE Float16/Float32 conversion at each source stage. Fixture exponent span makes product+acc exact inFloat64 beforeRNE; halfresultcuBLAS and rounded-productCUB semantics distinguished.',rounding_scope='Signed dyadic normal/subnormal products and original uniform generator value, finite only; noNaN/Inf/saturating arithmetic claims.',numpy=np.__version__,python=sys.version,generator=dict(path='src/f16sp-julia/generate_oracle.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())),indent=2)+'\n');print(json.dumps(cases))
