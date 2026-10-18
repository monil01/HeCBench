"""Independent intersecting-cell mean geometry plus clamp/crop and tile-footprint checks."""
from pathlib import Path
from fractions import Fraction
import json,struct,sys,math,hashlib
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());prefix=[];expected=[];stats=[]
for N,C,H,W in [(2,3,17,19),(1,2,32,33)]:
 for fixture in range(2):
  for mode in(1,2):
   zh,zw=(f32(1.5),f32(2.5))if mode==1 else(f32(.6),f32(.9));Ho=math.floor(f32(H*zh));Wo=math.floor(f32(W*zw));hs=abs(H-Ho)//2;ws=abs(W-Wo)//2;he=H+hs if mode==1 else H-Ho-hs;we=W+ws if mode==1 else W-Wo-ws;rh=f32(H/Ho);rw=f32(W/Wo);slots=(math.ceil(f32(16*H/Ho))+1)*(math.ceil(f32(16*W/Wo))+1);x=[((i*17)%97-48)/16 if fixture==0 else 1. for i in range(N*C*H*W)];prefix.extend(x);prefix.extend([N,C,H,W,Ho,Wo,H*W,hs,he,ws,we,mode,4*slots,3,zh,zw]);output=[];footprints=[];areas=[]
   for by in range((Ho+15)//16):
    for bx in range((Wo+15)//16):
     footprint=(math.ceil(f32((by+1)*16*rh))-math.floor(f32(by*16*rh)))*(math.ceil(f32((bx+1)*16*rw))-math.floor(f32(bx*16*rw)));assert footprint<=slots;footprints.append(footprint)
   for b in range(N*C):
    for h in range(H):
     for w in range(W):
      oh=h+hs if mode==1 else min(max(h-hs,0),Ho-1);ow=w+ws if mode==1 else min(max(w-ws,0),Wo-1);lh=Fraction(f32(oh*rh));eh=Fraction(f32((oh+1)*rh));lw=Fraction(f32(ow*rw));ew=Fraction(f32((ow+1)*rw));rows=[q for q in range(-1,H+2)if Fraction(q)<eh and Fraction(q+1)>lh];cols=[q for q in range(-1,W+2)if Fraction(q)<ew and Fraction(q+1)>lw];area=len(rows)*len(cols);assert area>0;v=sum(Fraction(x[b*H*W+i*W+j])for i in rows for j in cols if 0<=i<H and 0<=j<W)/area;output.append(f32(float(v)));areas.append(area)
   expected.extend(output);stats.append({'shape':[N,C,H,W],'fixture':'signeddyadic'if fixture==0 else'constantone','mode':'in'if mode==1 else'out','intermediate_zoom_shape':[Ho,Wo],'shared_slots':slots,'max_source_tile_footprint':max(footprints),'box_pixel_counts':sorted(set(areas)),'constant_field_preserved':all(v==1 for v in output)if fixture==1 else None,'output_values':len(output)})
allvalues=prefix+expected;assert len(prefix)==16328 and len(expected)==16200;reports=[]
for m,v in r['models'].items():
 for tr in v['trials']:
  p=Path(tr['output_artifact']['saved_path']);raw=p.read_bytes();got=struct.unpack('<32528f',raw);assert list(got[:16328])==prefix;err=max(abs(a-b)for a,b in zip(got[16328:],expected));assert err<=1e-4,(m,tr['trial'],err);reports.append({'model':m,'trial':tr['trial'],'full_output_pixels':16200,'exact_prefix_values':16328,'max_abs_error_to_independent_rounded_rational_mean':err,'sha256':hashlib.sha256(raw).hexdigest()})
w={'benchmark':'zoom','result':str(rp),'captured_trials_checked':len(reports),'method':'Independent unit-source-cell intersection predicate against actual Float32 mapping intervals, exact rational unweighted means, direct centered-crop/clamp coordinate map. No shared staging/reduction code or copied native CPU loops.','parameter_precision':'Float32 factors, dimension products, ratios and interval endpoints; rational mean then roundedFloat32.','scope':'Unweighted whole-pixel rectangle means, not bilinear or fractional-area weighting. Fixed input-sized output after crop/pad. Both source factors and two named shapes; huge shrink/resource fallback unmeasured.','cases':stats,'reports':reports};dest=Path('.porting-state/portable_julia/scientific_witnesses_300/zoom.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w))
