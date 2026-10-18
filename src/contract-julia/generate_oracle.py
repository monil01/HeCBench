"""Independent einsum contraction oracle, not a translation of source loop nests."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
p=Path(__file__).resolve().parent;N=9;C=18
idx=np.indices((N,N,N,C));a,b,c,f=idx
tensor=(((3*a+5*b+7*c+11*f+a*b+2*b*c+f*c)%33)-16).astype(np.float64)/16
x,y=np.indices((N,N));adj=(((5*x+3*y+x*y)%13)-6).astype(np.float64)/8
pos=np.maximum(adj,0)
expr=['abcf,de->abf','abcf,de->adf','abcf,de->bcf','abcf,de->bdf','abcf,de->def','abcf,ce->abf','abcf,dd->abf','abbf,de->adf','aecf,de->adf','dbcf,de->bcf','abaf,de->bdf','ebcf,de->bdf','abef,de->bdf','aacf,de->def','abbf,de->def','aeef,de->adf','ebef,de->bdf','aaaf,de->def']
outputs=np.stack([np.einsum(e,tensor,adj if i==5 else pos,optimize=False) for i,e in enumerate(expr)],axis=2)
assert outputs.shape==(N,N,18,C)
files={}
for name,data in [('study_tensor_f64.bin',tensor),('study_adj_f64.bin',adj),('oracle_f64.bin',outputs)]:
 raw=data.astype('<f8').tobytes();(p/name).write_bytes(raw);files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),count=data.size,bytes=len(raw),shape=list(data.shape))
meta=dict(generator='generate_oracle.py',numpy=np.__version__,python=sys.version,N=N,C=C,expressions=expr,case6_uses_signed_adj=True,other_cases_use_positive_adj=True,adj_counts=dict(positive=int((adj>0).sum()),negative=int((adj<0).sum()),zero=int((adj==0).sum())),tensor_distinct=len(np.unique(tensor)),all_intermediate_sums_exact_dyadic='Products multiples1/128, abs running sum<=729*0.75=546.75; integers scaled128<2^24, exactFloat32/Float64 without overflow.',files=files)
(p/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta))
