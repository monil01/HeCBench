"""Independent Float64 matrix oracle, fixed source six modes, seeded dyadic fixtures."""
from pathlib import Path
import numpy as np,json,hashlib
p=Path(__file__).resolve().parent;r=np.random.default_rng(20261018)
BT=np.array([[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],[0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]],dtype=float)
AT=np.array([[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]],dtype=float)
G=np.array([[.25,0,0],[-1/6,-1/6,-1/6],[-1/6,1/6,-1/6],[1/24,1/12,1/6],[1/24,-1/12,1/6],[0,0,1]],dtype=float)
fixture=[];gold=[];cases=[];guard=-12345.;aud=[]
def vals(shape,den=32):return (r.integers(-8,9,size=shape)/den).astype(np.float32)
for mode,(ci,co) in enumerate([(128,128),(256,256),(512,128),(128,512),(1024,256),(256,1024)]):
 bias=vals(co,32);scale=vals(co,8)
 if mode<2:
  a0=vals((16,16,ci),16);a=np.concatenate([a0.reshape(-1,ci),np.zeros((34,ci),dtype=np.float32)]);w=vals((ci,co,3,3),256)
  wt=np.einsum('ai,coij,bj->abco',G,w.astype(float),G).astype(np.float32)
  t=np.empty((6,6,16,ci),dtype=float)
  for x in range(4):
   for y in range(4):t[:,:,x*4+y,:]=np.einsum('ai,ijc,bj->abc',BT,np.array([[a[(x*4+i)*16+y*4+j] for j in range(6)] for i in range(6)],dtype=float),BT)
  ip=np.einsum('abtc,abco->abto',t,wt.astype(float))
  out=np.zeros((16,16,co),dtype=float)
  for x in range(4):
   for y in range(4):
    z=np.einsum('ai,ijo,bj->abo',AT,ip[:,:,x*4+y],AT)
    h=min(4,14-x*4);ww=min(4,14-y*4)
    out[1+x*4:1+x*4+h,1+y*4:1+y*4+ww]=np.maximum(z[:h,:ww]*scale+bias,0)
  direct=np.zeros_like(out)
  for i in range(14):
   for j in range(14):direct[i+1,j+1]=np.maximum(np.einsum('ijc,coij->o',a0[i:i+3,j:j+3].astype(float),w.astype(float))*scale+bias,0)
  aud.append({'mode':mode,'Float64_transformed_vs_direct_convolution_max_abs':float(np.max(abs(out-direct))),'random_nonzero_padded_input':True,'physical_halo_tail_vectors':34,'source_stride':16})
  arrays=[a,wt,bias,scale];outs=[t,ip,out]
 else:
  a=vals((196,ci),16);w=vals((ci,co),128);arrays=[a,w,bias,scale]
  out=a.astype(float)@w.astype(float);out=out*scale+bias
  if mode in (2,4):out=np.maximum(out,0)
  outs=[out]
 for arr in arrays:fixture.append(arr.ravel())
 for arr in outs:gold.append(np.r_[arr.ravel(),np.full(17,guard)])
 cases.append({'mode':mode,'ci':ci,'co':co,'input_sizes':[int(x.size) for x in arrays],'output_sizes':[int(x.size) for x in outs],'relu':mode not in (3,5)})
f=np.concatenate(fixture).astype('<f4');g=np.concatenate(gold).astype('<f8');f.tofile(p/'fixture.bin');g.tofile(p/'expected_outputs.bin')
(p/'fixture_description.json').write_text(json.dumps({'seed':20261018,'cases':cases,'prefix_count':len(f),'output_count':len(g),'independent_oracles':aud,'input_sha256':hashlib.sha256(f.tobytes()).hexdigest(),'gold_sha256':hashlib.sha256(g.tobytes()).hexdigest(),'oracle_note':'Float64 matrix transforms and GEMM; supplementary direct spatial 3x3 convolution checks independently validate transformed weight convention. Bias loaded by original mode0 but unused is omitted. Original input random padding retained.'},indent=2)+'\n')
