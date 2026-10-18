from pathlib import Path
import struct,json,hashlib,math
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
# Independent Float64 filtering algebra + exact signed32 RNG state; retain source truncation and normal rather than uniform resampling offset.
seed=lambda z: ((1103515245*z+12345+(1<<31))%(1<<32)-(1<<31))
def advance(z):
 v=seed(z);return v%2147483647 if v>=0 else -((-v)%2147483647)
def normal(z):
 z=advance(z);u=abs(float(z)/2147483647);z=advance(z);v=abs(float(z)/2147483647);return z,math.sqrt(-2*math.log(u))*math.cos(2*math.pi*v)
obj=[(dy,dx)for dx in range(-4,5)for dy in range(-4,5)if dx*dx+dy*dy<25];assert len(obj)==69
video=bytes(224+((x+2*y+3*k)%5)for x in range(64)for y in range(64)for k in range(4));(B/'video_u8.bin').write_bytes(video);(B/'offsets_i32.bin').write_bytes(struct.pack('<'+'i'*(2*len(obj)),*(v for pair in obj for v in pair)))
oracle=[];descs=[];prefix=[];state=[];min_index_margin=1.;min_cdf_margin=1.
for n in [17,256,513]:
 seeds=list(range(1,n+1));xs=[32.125+(i%7)/16 for i in range(n)];ys=[31.375-(i%5)/32 for i in range(n)];case=[]
 for frame in range(1,4):
  ax=[];ay=[];ind=[];like=[];weights=[]
  for i in range(n):
   seeds[i],zx=normal(seeds[i]);seeds[i],zy=normal(seeds[i]);x=xs[i]+1+5*zx;y=ys[i]-2+2*zy;ax.append(x);ay.append(y);min_index_margin=min(min_index_margin,abs(x-math.trunc(x)),abs(y-math.trunc(y)),1-abs(x-math.trunc(x)),1-abs(y-math.trunc(y)));idxs=[]
   for dy,dx in obj:
    idx=abs((math.trunc(x)+dx)*64*4+(math.trunc(y)+dy)*4+frame);idx=idx if idx<len(video)else 0;idxs.append(idx)
   ind.extend(idxs);ll=sum(((video[j]-100)**2-(video[j]-228)**2)/50 for j in idxs)/69-300;like.append(ll);weights.append(math.exp(ll)/n)
  total=sum(weights);nw=[z/total for z in weights];cdf=[];s=0
  for z in nw:s+=z;cdf.append(s)
  seeds[0],off=normal(seeds[0]);u=[off/n+i/n for i in range(n)];select=[]
  for threshold in u:
   ix=next((j for j,z in enumerate(cdf)if z>=threshold),n-1);select.append(ix);min_cdf_margin=min(min_cdf_margin,min(abs(z-threshold)for z in cdf))
  xs=[ax[j]for j in select];ys=[ay[j]for j in select]
  # Full source buffers in declared order; partial sums onlysourcewrittenblocks, plusfullintindices/seeds.
  parts=[sum(weights[j:j+256])for j in range(0,n,256)];parts[0]=total
  arrays=[ax,ay,like,nw,cdf,u,xs,ys,parts,ind,seeds]
  case.append(dict(frame=frame,oracle_offset=len(oracle),lengths=list(map(len,arrays)),selected_indices=select));oracle.extend(v for a in arrays for v in a)
 descs.append(dict(n=n,frames=case))
(B/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(oracle),*oracle));(B/'cases.json').write_text(json.dumps(dict(cases=descs,index_margin=min_index_margin,cdf_margin=min_cdf_margin,scope='IndependentFloat64algebra engineeringoracle, notroundedsourceoperationemulation; exactsigned32wrapsourcepolicy RNG, actualinputFloat32seed/Mratio insteadhigherprecision can differseveraltinyulps, predeclaredtypewiseerrors. ExactindicesandRNGcheckedseparately; no sourcevideo generator/predictionaccuracyclaim'),indent=2)+'\n');print('oracle',len(oracle),'margins',min_index_margin,min_cdf_margin)
