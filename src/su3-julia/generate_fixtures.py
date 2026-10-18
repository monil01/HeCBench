from pathlib import Path
import json,hashlib,sys
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[1];W=R/'.porting-state/portable_julia/su3_reference_review';assert sys.byteorder=='little'
G=np.uint32(0xa5a5a5a5);specs=[(1,36,0,1),(16,128,1,2),(17,128,2,4),(33,64,3,4),(257,64,4,8),(4096,128,2,8),(1048576,128,1,32)]
inputs=[];oracles=[];cases=[];io=oo=0

def pack(a):return np.ascontiguousarray(a,dtype='<f4').view('<u4')
def determinant(z):return z[0,0]*(z[1,1]*z[2,2]-z[1,2]*z[2,1])-z[0,1]*(z[1,0]*z[2,2]-z[1,2]*z[2,0])+z[0,2]*(z[1,0]*z[2,1]-z[1,1]*z[2,0])
def monomial(shift,r):
 z=np.zeros((3,3),dtype=np.complex128);phase=[1,1j,-1,-1j]
 for row,e in enumerate([r,(-r)%4,0]):z[row,(row+shift)%3]=phase[e%4]
 assert np.array_equal(z@z.conj().T,np.eye(3))and determinant(z)==1
 return z
for cid,(n,th,mode,ldim)in enumerate(specs):
 a=np.full((n,80),G,dtype='<u4');out=np.full_like(a,G);index=np.arange(n,dtype='<u4');a[:,72]=index%ldim;a[:,73]=(index//ldim)%ldim;a[:,74]=(index//ldim**2)%ldim;a[:,75]=(index//ldim**3)%ldim;a[:,76]=index;a[:,77]=(G&np.uint32(0xffffff00))|np.where((a[:,72]+a[:,73]+a[:,74]+a[:,75])%2==0,2,1).astype('<u4')
 av=a[:,:72].reshape(n,4,3,3,2);bv=np.zeros((4,3,3,2),dtype='<f4');expected=np.zeros((n,4,3,3,2),dtype='<f4');groupchecks=0;bound=0
 if mode==1:
  va=np.zeros((n,4,3,3,2),dtype='<f4');va[...,0]=1;av[:]=pack(va);bv[...,0]=np.float32(1/3);expected[...,0]=1
  assert np.float32(3*float(np.float32(1/3)))==1
 elif mode==4:
  ac=np.empty((n,4,3,3),dtype=np.complex128);bc=np.empty((4,3,3),dtype=np.complex128)
  for direction in range(4):
   bc[direction]=monomial((direction+1)%3,(direction+2)%4)
   for site in range(n):ac[site,direction]=monomial((site+direction)%3,(site*3+direction)%4)
  cc=ac@bc
  for z in cc.reshape(-1,3,3):assert np.array_equal(z@z.conj().T,np.eye(3))and determinant(z)==1;groupchecks+=1
  aa=np.stack((ac.real,ac.imag),axis=-1).astype('<f4');bv=np.stack((bc.real,bc.imag),axis=-1).astype('<f4');av[:]=pack(aa);expected=np.stack((cc.real,cc.imag),axis=-1).astype('<f4')
 else:
  t=np.arange(n*72,dtype=np.int64).reshape(n,4,3,3,2);au=(t*7+cid*3)%17-8;bu=(np.arange(72,dtype=np.int64).reshape(4,3,3,2)*5+cid*11)%19-9
  if mode==0:au[:]=0
  if mode==3:
   bu[:]=0
   for direction in range(4):
    for k in range(3):bu[direction,k,k,0]=16
  aa=(au/8).astype('<f4');bv=(bu/16).astype('<f4');av[:]=pack(aa)
  real=np.einsum('sdik,dkj->sdij',au[...,0],bu[...,0])-np.einsum('sdik,dkj->sdij',au[...,1],bu[...,1]);imag=np.einsum('sdik,dkj->sdij',au[...,0],bu[...,1])+np.einsum('sdik,dkj->sdij',au[...,1],bu[...,0]);bound=6*int(np.abs(au).max())*int(np.abs(bu).max());assert bound<2**24
  expected=np.stack((real/128,imag/128),axis=-1).astype('<f4')
  # Separately evaluate full complex matrix products in binary64 (exactdyadic).
  cc=(aa[...,0].astype(np.float64)+1j*aa[...,1])@(bv[...,0].astype(np.float64)+1j*bv[...,1]);assert np.array_equal(cc.real,expected[...,0])and np.array_equal(cc.imag,expected[...,1])
  if mode==3:assert np.array_equal(expected,aa)
 out[:,:72]=pack(expected).reshape(n,72)
 if cid==1:
  assert a.tobytes()==(W/'initializer_sites.bin').read_bytes();assert bv.tobytes()==(W/'initializer_B.bin').read_bytes()
 case=dict(case_id=cid,n=n,threads=th,mode=mode,ldim=ldim,io=io,oo=oo,source_full_shape=n==1048576,source_initializer=mode==1,special_unitary_product_matrices_checked=groupchecks,max_abs_complex_partial_sum_units_1_over128_bound=bound);cases.append(case);inputs.extend([a.reshape(-1),pack(bv).reshape(-1)]);oracles.append(out.reshape(-1));io+=80*n+72;oo+=80*n
with(B/'input_u32.bin').open('wb')as f:
 for a in inputs:a.tofile(f)
with(B/'oracle_u32.bin').open('wb')as f:
 for a in oracles:a.tofile(f)
info=dict(cases=cases,site_bytes=320,matrix_bytes=72,site_words=80,link_words=72,metadata_policy='ActualCPUABI/sourceinitializer confirmed; sourcepadding/unusedCmetadata explicitlypoisoninitialized0xa5, allfullAoSwordschecked. CPUoracleonlyhost, neverGPUupload.',scope='OriginalfullL32/sites1048576/blocks=N/128threads/rankone1and1/3 initializer pluszero,identity,densedyadiccomplexandspecialunitarymonomialcontrols. NamedR3/source1000unmeasured. OddsitesoperatorcontrolsnotfullcubicCLI. Sourceconstantmatrices arenotunitarygroup elements; operatornotfullQCDdynamics.',generation=dict(numpy_version=np.__version__,python=sys.version,CPU_initializer_layout=json.loads((W/'initializer_layout.json').read_text()),command='PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/su3-julia/generate_fixtures.py'))
info['large_local_only_assets']=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())for p in [B/'input_u32.bin',B/'oracle_u32.bin']];(B/'cases.json').write_text(json.dumps(info,indent=2)+'\n');keys=['n','threads','mode','ldim','io','oo'];(B/'study_cases.h').write_text('struct Case{int '+','.join(keys)+';};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in keys)+'}'for c in cases)+'};\n');(B/'study_cases.jl').write_text('const CASES=['+','.join('('+','.join(k+'='+str(c[k])for k in keys)+',)'for c in cases)+']\n');print(info['large_local_only_assets'])
