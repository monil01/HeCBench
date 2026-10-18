from pathlib import Path
from decimal import Decimal,getcontext
import struct,json
getcontext().prec=80;p=Path(__file__).resolve().parent;F=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];D=lambda x:Decimal.from_float(float(x));eps=F(1e-5)
prefix=[6];gold=[];cases=[];paths=[];G=-12345.
for N,C in [(5,256),(7,259),(3,1024)]:
 for mode in [0,1]:
  inp=[F(((r*11+c*7)%53-26)/32 if mode==0 else (r-2)/16) for r in range(N) for c in range(C)];weight=[F(0 if c%19==0 else ((c*3)%17-8)/8) for c in range(C)];bias=[F(((c*5)%23-11)/16) for c in range(C)];prefix+=[N,C,mode];prefix+=inp+weight+bias
  means=[];rs=[];out=[]
  for r in range(N):
   x=list(map(D,inp[r*C:(r+1)*C]));m=sum(x)/C;v=sum((a-m)**2 for a in x)/C;s=1/(v+D(eps)).sqrt();means.append(float(m));rs.append(float(s));out.extend(float((a-m)*s*D(w)+D(b)) for a,w,b in zip(x,weight,bias))
  ci=len(cases);cases.append(dict(case=ci,N=N,C=C,mode='bounded_textured' if mode==0 else 'exact_constant_rows',U=4 if C%4==0 else 1,output_sizes=[N*C,N,N],input_sizes=[N*C,C,C]))
  for algo in range(3):
   for BT in [32,64,128,256,512,1024]:
    for a in [out,means,rs]:gold +=[G]*8+a+[G]*8
    for a in [inp,weight,bias]:gold +=[G]*8+a+[G]*8
    paths.append(dict(case=ci,algorithm=algo,threads=BT,unroll=4 if algo==0 and C%4==0 else 1,grid=N if algo==0 else (N*32+BT-1)//BT))
for name,arr in [('fixture.bin',prefix),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(arr),*arr))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,paths=paths,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),oracle='IndependentDecimal80 mean/centralvariance/affineoutput plusactualFloat32eps; kernel2momentidentity comparedwithsame mathematicalcentralvariance budget',input_precision='ActualFloat32valueslosslesslywidenedFloat64',eps_float32=eps,atol=1e-5,rtol=2e-6,support='Boundedtextured andexactconstantrows, nonzero/zero/negativegamma, variedbeta; no cancellation-pronehugeoffsetaccuracyclaim'),indent=2)+'\n')
