"""Independent exact integer convolution via padded sliding windows/einsum."""
from pathlib import Path
import numpy as np,json,hashlib,sys
P=Path(__file__).resolve().parent;N,C,H,W=2,3,11,13;n,c,y,x=np.indices((N,C,H,W));image=((7*n+11*c+3*y+5*x+y*x)%33-16).astype(np.int64);pad=np.pad(image,((0,0),(0,0),(1,1),(1,1)));cases=[];files={}
def save(name,data):
 raw=np.asarray(data,dtype='<f4').tobytes();(P/name).write_bytes(raw);files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),count=np.asarray(data).size)
ci=0
for m in range(1,5):
 for k in [1,3,5]:
  ci+=1;oc=C*m;co,ky,kx=np.indices((oc,k,k));weights=((13*co+7*ky+3*kx+ci)%17-8).astype(np.int64);bias=((np.arange(oc)+ci)%9-4).astype(np.int64);oh=H+3-k;ow=W+3-k;out=np.empty((N,oc,oh,ow),dtype=np.int64)
  for o in range(oc):
   windows=np.lib.stride_tricks.sliding_window_view(pad[:,o//m],(k,k),axis=(-2,-1));out[:,o]=np.einsum('nhwij,ij->nhw',windows,weights[o],optimize=False)+8*bias[o]
  fixture=np.concatenate([image.ravel()/16,weights.ravel()/16,bias.ravel()/32]);save(f'study_case{ci:02}_f32.bin',fixture);save(f'oracle_case{ci:02}_f32.bin',out/256)
  params=[m,N,C,H,W,k,oh,ow,1,1,1,1,1,1,1,N*oc*oh*ow,10,k if k in [1,3] else 0]
  cases.append(dict(case=ci,m=m,k=k,n=N,c=C,h=H,w=W,oc=oc,oh=oh,ow=ow,input_count=image.size,weight_count=weights.size,bias_count=bias.size,output_count=out.size,params=params))
meta=dict(cases=cases,files=files,input_denominator=16,weight_denominator=16,bias_denominator=32,oracle_denominator=256,oracle='IndependentexactInt64paddedslidingwindows/einsum,oc//mchannelmap; no floatingkernelloopindextranscription',precision_proof='Productsandallpartialsumsareintegerscaled1/256,absaccum<=25*.5+.125<13,scaledmagnitude<3328<2^24; Float32intermediatesexact,no overflow.',numpy=np.__version__,python=sys.version,generator='generate_oracle.py',atol=0,rtol=0)
meta['prefix_count']=sum(x['input_count']+x['weight_count']+x['bias_count']+len(x['params']) for x in cases);meta['output_count']=sum(x['output_count'] for x in cases);meta['guard_count']=len(cases)*4*32;meta['artifact_count']=meta['prefix_count']+meta['output_count']+meta['guard_count'];(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:meta[k] for k in ['prefix_count','output_count','guard_count','artifact_count']})
