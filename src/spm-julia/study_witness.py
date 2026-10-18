"""Independent eight-corner barycentric check; never changes acceptance/timing captures."""
from pathlib import Path
import array,struct,math,json,re,sys,hashlib
ROOT=Path(__file__).resolve().parents[2]
F=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
RAN=[F(float(x)) for x in re.findall(r'Float32\(([^)]+)\)',(Path(__file__).parent/'study_constants.jl').read_text())]
U=2**-24
BOUND=8*U/(1-8*U)*250.5+32*2**-52*251

def read(p):
 a=array.array('f');a.frombytes(Path(p).read_bytes());return a

def expected(a):
 pos=0;cases=[]
 for v in [17,33]:
  for mode in [0,1]:
   vv,mm,n,size,reps=map(int,a[pos:pos+5]);pos+=5;assert(vv,mm,n,size,reps)==(v,mode,v**3,(v+1)**2*(v+5),3)
   M=a[pos:pos+16];pos+=16;g=a[pos:pos+size];pos+=size;f=a[pos:pos+size];pos+=size
   assert list(g)==[(i*17+i//v*7)%251 for i in range(size)]
   assert list(f)==[(i*29+i//(v*v)*13)%251 for i in range(size)]
   cases.append((v,mode,n,g,f,M))
 prefix=pos;checks=[];amb=[];valid=0
 for v,mode,n,g,f,M in cases:
  qa=[];qb=[];qt=[]
  for i in range(n):
   r=RAN[i%97];rx=F(F(i%(v-2)+1)+r);ry=F(F((i//(v-2))%(v-2)+1)+r);rz=F(F(F(math.floor(F(F(i)/F(v-2)))/F(v-2))+1)+r)
   xp=F(rx+M[12]);yp=F(ry+M[13]);zp=F(rz+M[14]);t=1<=xp<v and 1<=yp<v and 1<=zp<v;qt.append((int(t),int(t),float(t)))
   if not t:qa.append((0,0,0.));qb.append((0,0,0.));continue
   valid+=1
   for name,vol,x,y,z,out in [('f',f,xp,yp,zp,qa),('g',g,rx,ry,rz,qb)]:
    ix,iy,iz=map(math.floor,[x,y,z]);dx=x-ix;dy=y-iy;dz=z-iz;s=0.;corners=[];weights=[]
    # Weight subtractions are exact here: input coordinates are Float32 >=1,
    # so fractions and 1-fractions have at most23 fractional binary places.
    assert F(dx)==dx and F(1-dx)==1-dx and F(dy)==dy and F(1-dy)==1-dy and F(dz)==dz and F(1-dz)==1-dz
    for oz in [0,1]:
     for oy in [0,1]:
      for ox in [0,1]:
       w=(dx if ox else 1-dx)*(dy if oy else 1-dy)*(dz if oz else 1-dz);k=vol[ix+ox-1+v*(iy+oy-1+v*(iz+oz-1))];s+=k*w;corners.append(k);weights.append(w)
    lo=math.floor(s-BOUND+.5);hi=math.floor(s+BOUND+.5);out.append((lo,hi,s))
    if lo!=hi:amb.append({'dimension':v,'transform_mode':mode,'index0':i,'field':name,'coordinates':[x,y,z],'corners_xyz_binary_order':corners,'weights_same_order':weights,'float64_interpolant':s,'error_bound':BOUND,'admissible_quantized_bins':[lo,hi],'distance_to_half_bin':abs(s-math.floor(s)-.5)})
  checks+=qa+qb+qt
 return prefix,checks,amb,valid

def main():
 result=Path(sys.argv[1]).resolve();d=json.loads(result.read_text());first=d['models']['cuda']['trials'][0]['output_artifact']['saved_path'];base=read(first);prefix,checks,amb,valid=expected(base);records=[];maxerr=0.;nondoublebins=[]
 for model,rec in d['models'].items():
  for trial in rec.get('trials',[]):
   p=Path(trial['output_artifact']['saved_path']);a=read(p);assert a[:prefix]==base[:prefix];assert len(a)==prefix+len(checks)
   for j,(q,(lo,hi,s)) in enumerate(zip(a[prefix:],checks)):
    assert q==int(q) and lo<=q<=hi,(model,trial['trial'],j,q,lo,hi,s)
    maxerr=max(maxerr,abs(q-s))
   records.append({'model':model,'trial':trial['trial'],'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checked_outputs':len(checks),'input_prefix_exact':True,'all_quantized_samples_in_interval_and_all_masks_exact':True})
 output={'benchmark':'spm','configuration_id':d['configuration_id'],'raw_result':str(result),'input_prefix_count':prefix,'output_count':len(checks),'valid_sample_pairs_per_capture':valid,'ambiguous_sample_values_per_capture':len(amb),'max_quantized_difference_from_float64_interpolant':maxerr,'fp32_interpolation_bound':BOUND,'bound_derivation':'All eight byte corners <=250; exact Float32 fractional coordinates/weights, nonnegative expression, at most6 rounded arithmetic stages along each corner contribution plus final +.5; gamma8 conservatively bounds these by gamma8*(250+.5), plus32eps64*251 for independent double evaluation. No overflow/underflow/cancellation. Outside the interval there is one allowed rounded bin; inside it only adjacent interval-admissible bins are allowed. Exact GPU-vs-GPU artifacts separately require tolerance0.','post_observation_revision':'Original strict nativeGPU-vs-nativeCPU one-byte rounding rejection retained in20261018T015105774443Z_spm. Auxiliary contract revision is explicit; never broadens dualGPU tolerance.','ambiguity_witnesses':amb,'captures':records,'pass':len(records)==20}
 out=Path(__file__).parent/'independent_witness.json';out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items()if k not in ['ambiguity_witnesses','captures']},indent=2))
if __name__=='__main__':main()
