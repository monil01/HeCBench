"""Independent Float64 area geometry and positive-distance weighted mean oracle."""
from pathlib import Path
import numpy as np,json,hashlib,math,sys
P=Path(__file__).resolve().parent;cases=[];files={}
def write(name,data):
 raw=np.asarray(data,dtype=np.uint8).tobytes();(P/name).write_bytes(raw);files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for ci,(iw,ih) in enumerate([(40,30),(128,96)],1):
 ow,oh=16,12;image=np.empty((ih,iw,3),dtype=np.uint8)
 for y in range(ih):
  for x in range(iw):
   image[y,x]=[40,80,120] if x<max(8,iw//4) and y<max(8,ih//4) else [(17*x+13*y+x*y%31+29*c+11*(x//3))%220+16 for c in range(3)]
 write(f'study_input{ci}_u8.bin',image);pw,ph=iw/ow,ih/oh;weights={};patches=np.empty((oh,ow,3),dtype=np.uint8)
 for y in range(oh):
  for x in range(ow):
   # Separable exact interval intersections, not source's sequential edge factors.
   xs=np.arange(math.floor(x*pw),math.ceil((x+1)*pw));ys=np.arange(math.floor(y*ph),math.ceil((y+1)*ph));wx=np.maximum(0,np.minimum(xs+1,(x+1)*pw)-np.maximum(xs,x*pw));wy=np.maximum(0,np.minimum(ys+1,(y+1)*ph)-np.maximum(ys,y*ph));w=wy[:,None]*wx[None,:];pixels=image[ys[:,None],xs[None,:]].astype(np.float64);mean=(pixels*w[:,:,None]).sum(axis=(0,1))/w.sum();patches[y,x]=np.floor(mean).astype(np.uint8);weights[y,x]=(pixels,w)
 write(f'oracle_guidance{ci}_u8.bin',patches);case=dict(case=ci,iw=iw,ih=ih,ow=ow,oh=oh,pw=pw,ph=ph,source_patch_samples='9fractionalor64integer',lambdas={})
 for lam,label in [(0,'0'),(1,'1'),(.5,'half')]:
  output=np.empty_like(patches);fallback=0
  for y in range(oh):
   for x in range(ow):
    ya,yb=max(0,y-1),min(oh,y+2);xa,xb=max(0,x-1),min(ow,x+2);wx=np.array([2 if v==x else 1 for v in range(xa,xb)]);wy=np.array([2 if v==y else 1 for v in range(ya,yb)]);gw=wy[:,None]*wx[None,:];avg=(patches[ya:yb,xa:xb].astype(np.float64)*gw[:,:,None]).sum(axis=(0,1))/gw.sum();pixels,area=weights[y,x];d=np.linalg.norm(pixels-avg,axis=2)
    # The common 1/sqrt(3*255^2)^lambda factor cancels exactly in normalizedratio.
    importance=np.ones_like(d) if lam==0 else d**lam;w=area*importance
    if w.sum()==0:mean=avg;fallback+=1
    else:mean=(pixels*w[:,:,None]).sum(axis=(0,1))/w.sum()
    output[y,x]=np.floor(mean).astype(np.uint8)
  write(f'oracle_output{ci}_lambda_{label}_u8.bin',output);case['lambdas'][label]=dict(value=lam,zero_weight_fallback_pixels=fallback)
 cases.append(case)
(P/'fixture_provenance.json').write_text(json.dumps(dict(generator='generate_oracle.py',numpy=np.__version__,python=sys.version,cases=cases,files=files,oracle='IndependentFloat64intervaloverlapgeometry,separableGaussianneighborweights,unnormalizedL2importancecommonfactorcancelled,quantizedguidance/output',predeclared_byte_atol=1,rtol=0,uniform_region='KnownconstantRGB40/80/120andzero-distancefallback exercised',limits=['AllRGBoutputscomparednotimage-quality/trainingaccuracy','NativewarporderedFloat32vsexistingJuliaserialFloat32 vsFloat64oracle maydifferby1byte;sourcevalidatoralsoboundsroundingdifference1','Original8192squareimagecache/workloadnotcertified']),indent=2)+'\n');print(cases)
