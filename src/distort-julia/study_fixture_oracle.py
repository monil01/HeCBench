from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
u=2**-24;tiny=2**-150

def shift(a,b,c,k,dt=1.):
 for iteration in range(100):
  mid=f(a+f(f(b-a)*f(.5)));q=f(a+f(f(f(a-c)*k)*f(f(a-c)*f(a-c))));q3=f(mid+f(f(f(mid-c)*k)*f(f(mid-c)*f(mid-c))))
  if -dt<q<dt:return a
  if q3<0:a=mid
  else:b=mid
 raise ValueError('NativeShiftIterationLimit')

def expand(lo,hi):
 e=u*max(abs(lo),abs(hi))+tiny
 return math.nextafter(lo-e,-math.inf),math.nextafter(hi+e,math.inf)
def add(a,b):return expand(a[0]+b[0],a[1]+b[1])
def neg(a):return -a[1],-a[0]
def sub(a,b):return add(a,neg(b))
def mul(a,b):
 v=[x*y for x in a for y in b];return expand(min(v),max(v))
def pt(a):return a,a

def coord(x,y,P):
 k,cx,cy,w,h,th,xs,ys,xsh,ysh=P;sx=add(mul(pt(x),pt(xs)),pt(xsh));sy=add(mul(pt(y),pt(ys)),pt(ysh));dx=sub(sx,pt(cx));dy=sub(sy,pt(cy));r2=add(mul(dx,dx),mul(dy,dy));return add(sx,mul(mul(dx,pt(k)),r2)),add(sy,mul(mul(dy,pt(k)),r2))

cases=[(17,19,0.,0),(33,31,2**-10,1),(32,16,2**-14,2),(19,17,2**-13,3)];fixture=[];lower=[];upper=[];witness=[];reviews=[];maxerr=0.;gamma=(16*u)/(1-16*u)
for ci,(w,h,k,mode)in enumerate(cases):
 k=f(k);cx=f(w//2);cy=f(h//2);xsh=shift(0.,f(cx-1),cx,k);xsh2=shift(0.,f(f(w)-cx-1),f(f(w)-cx),k);ysh=shift(0.,f(cy-1),cy,k);ysh2=shift(0.,f(f(h)-cy-1),f(f(h)-cy),k);xs=f(f(f(w)-xsh-xsh2)/f(w));ys=f(f(f(h)-ysh-ysh2)/f(h));P=[k,cx,cy,w,h,1.,xs,ys,xsh,ysh];Pbits=[struct.unpack('<I',struct.pack('<f',x))[0]if j not in [3,4]else x for j,x in enumerate(P)];data=[]
 for r in range(h):
  for c in range(w):
   for ch in range(3):data.append(((c*73+r*151+ch*47+c*r*17)^((c+11)*(r+7)))%256 if mode==0 else (c*(3+ch)+r*(5-ch)+ch*31)%256 if mode==1 else 255*((c+r+ch)%2)if mode==2 else 123)
 fixture +=[w,h,mode,5]+Pbits+data;ambiguous=0;inside=0;outside=0;maxcase=0.
 for y in range(h):
  for x in range(w):
   if k==0:
    for ch in range(3):lower.append(data[3*(y*w+x)+ch]);upper.append(lower[-1])
    inside+=1;continue
   xi,yi=coord(x,y,P)
   if xi[1]<0 or yi[1]<0 or xi[0]>w-1 or yi[0]>h-1:
    lower.extend([0]*3);upper.extend([0]*3);outside+=1;continue
   if not(xi[0]>=0 and yi[0]>=0 and xi[1]<=w-1 and yi[1]<=h-1):raise ValueError(('PreGPUFixtureMaskUncertainty',ci,x,y,xi,yi))
   inside+=1;sx=x*xs+xsh;sy=y*ys+ysh;dx=sx-cx;dy=sy-cy;xn=sx+dx*k*(dx*dx+dy*dy);yn=sy+dy*k*(dx*dx+dy*dy);x0=math.floor(xn);x1=math.ceil(xn);y0=math.floor(yn);y1=math.ceil(yn);a=yn-y0;b=xn-x0;weights=[(1-a)*(1-b),(1-a)*b,a*b,a*(1-b)];corners=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)];ex=max(abs(xi[0]-xn),abs(xi[1]-xn));ey=max(abs(yi[0]-yn),abs(yi[1]-yn));err=255*(ex+ey)+gamma*255+2**-40;maxcase=max(maxcase,err);maxerr=max(maxerr,err)
   for ch in range(3):
    vals=[data[3*(rr*w+cc)+ch]for cc,rr in corners];v=math.fsum(z*t for z,t in zip(vals,weights));lo=max(0,min(255,math.floor(v-err)));hi=max(0,min(255,math.floor(v+err)));assert 0<=hi-lo<=1,('OracleIntervalTooWide',ci,x,y,ch,v,err,lo,hi);lower.append(lo);upper.append(hi)
    if lo!=hi:ambiguous+=1;witness.append({'case':ci,'x':x,'y':y,'channel':ch,'four_corners':corners,'corner_values':vals,'Float64_weights':weights,'nominal_value':v,'coordinate_intervals':[xi,yi],'derived_abs_error':err,'admissible_bins':[lo,hi]})
 reviews.append({'width':w,'height':h,'K':k,'mode':mode,'properties_bits':Pbits,'strictly_classified_inside_pixels':inside,'strictly_classified_outside_pixels':outside,'uncertain_masks':0,'adjacent_bin_channels':ambiguous,'max_interpolation_bound':maxcase})
(p/'fixtures_u32.bin').write_bytes(struct.pack('<'+'I'*len(fixture),*fixture));(p/'oracle_lower_u8.bin').write_bytes(bytes(lower));(p/'oracle_upper_u8.bin').write_bytes(bytes(upper));(p/'oracle_interval_witnesses.json').write_text(json.dumps(witness,indent=2)+'\n');(p/'independent_oracle.json').write_text(json.dumps({'method':'Float64 sourceparameter polynomialwarp + independent bilinear fourcorner weights, with outwardintervals containing exactreal andeachFP32rounding step includingcontractedpaths; coordinate Lipschitz255*(ex+ey) andgamma16*255 interpolation envelope','rounding_unit':u,'rounding_floor':tiny,'gamma16':gamma,'bound_explanation':'Everybasicoperation retains exactreal interval and expands byu*maxabs+2^-150 withFloat64outwardnextafter, so bothrounded andfused paths included. Bilinear image is piecewisecontinuous andLipschitz<=255 peraxis globally.16roundingops conservativelycover4weighted3-products+sum onnonnegative weights. Source truncates positivefloattoUInt8, requireflooredintegerwithinperchannelbound; ifintervaldoesnotcrossinteger requireexactbin, otherwiseonlytwoadjacentbins. Maskclassification mustbeunambiguous beforeGPU.','cases':reviews,'maximum_derived_interpolation_abs_error':maxerr,'ambiguous_bin_channels':len(witness),'witness_path':'oracle_interval_witnesses.json','limits':['Quantizedsourcebarrelwarp notphysicalcamera calibration','Originalautomaticpropertypreprocessing checkedseparately; givenexactFloat32properties are mathematicalinputs','DualGPUcomparisonsatol1byte onlybutindependentperchannelintervalusuallyunique, no blanketonebyteoracle waiver','FP32coordinate/radialpathlowering can differFMA, sourcetruncationcell differences expected']},indent=2)+'\n');print({'prefix':len(fixture),'outputs':len(lower),'adjacent_channels':len(witness),'maxbound':maxerr,'reviews':reviews})
