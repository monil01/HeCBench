"""Explicit fixture regeneration; never run while verification is queued/running."""
import pathlib,struct,math,json
r=pathlib.Path(__file__).resolve().parents[2];b=pathlib.Path(__file__).resolve().parent;f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];focals=list(map(f32,[.005,.02,.036]));surfs=[];descriptions=[]
def add(pixel,findex,scale,normal=(0,0,1),shift=0,r2=1/64,label=''):
 x,y=pixel;px=(x-15.5)*scale+shift;py=(y-15.5)*scale;pz=f32(focals[findex]*scale);surf=list(map(f32,[px,py,pz,*normal,r2]));surfs.append(surf);descriptions.append({'index':len(surfs)-1,'target_pixel':pixel,'focal_index':findex,'scale':scale,'label':label})
pixels=[(2,2),(7,9),(13,15),(18,18),(24,27),(30,3)]
for fi in range(3):
 for j,p in enumerate(pixels):
  add(p,fi,64,(0,0,1)if j%2==0 else(.6,0,.8),label='foregrounddisk');add(p,fi,128,(.6,0,.8)if j%2==0 else(0,0,1),label='occludedfardisk')
add((9,25),0,64,shift=.125,label='strict_radius_equal_must_miss')
add((17,6),1,64,shift=.125,r2=.02,label='inside_radius')
add((26,14),2,64,shift=.125,r2=.01,label='outside_radius')
add((3,29),0,32768,label='depth_above100_clipped_zero')
while len(surfs)<256:surfs.append(list(map(f32,[1e6+len(surfs)*16,1e6,2,0,0,1,1/64])))
add((5,11),2,64,label='tail_surfel_index256_visible')
assert len(surfs)==257
maps=[];hits=[]
for f in focals:
 out=[]
 for y in range(32):
  for x in range(32):
   ray=(x-15.5,y-15.5,f);depth=1e20
   for surf in surfs:
    p=surf[:3];n=surf[3:6];dot=sum(a*v for a,v in zip(n,ray));alpha=sum(a*v for a,v in zip(n,p))/dot
    radial=sum((a*alpha-v)**2 for a,v in zip(ray,p));candidate=f*alpha
    if radial<surf[6]and candidate<depth:depth=candidate
   out.append(0. if depth>100 else depth)
 maps.extend(out);hits.append(sum(v>0 for v in out))
assert all(x>0 for x in hits)and all(x<1024 for x in hits)
assert maps[25*32+9]==0
assert maps[1024+6*32+17]>0
assert maps[2048+14*32+26]==0
assert maps[2048+11*32+5]>0
header='#include <cstdint>\n';jl=''
for name,values in [('SRC',[x for row in surfs for x in row]),('FOCALS',focals),('EXPECTED',list(map(f32,maps)))]:
 words=[struct.unpack('<I',struct.pack('<f',x))[0]for x in values];header+='static const uint32_t STUDY_'+name+'_BITS[]={'+','.join(hex(v)+'u'for v in words)+'};\n';jl+='const STUDY_'+name+'=reinterpret(Float32,UInt32['+','.join(hex(v)for v in words)+'])\n'
(b/'study_fixture.h').write_text(header);(b/'study_fixture.jl').write_text(jl)
missing=[y*16+x for y in range(16)for x in range(16)if x>=15 or y>=13]
meta={'generator':'Positive-depthfront/tiltedplaneunit-normaldiscsaimedatexactpower-of-two-scaledpixelrays;occlusion/strictdiskboundary/cutoff/tailwitnesses','surfel_count':257,'width':32,'height':32,'focals':focals,'nonzero_pixels_per_focal':hits,'max_unit_normal_norm2_error':max(abs(sum(v*v for v in s[3:6])-1)for s in surfs),'known_geometric_cases':descriptions,'independent_oracle':'Float64line-planeintersectionanddiskmembership,minimumdepth+original>100cutoffusingactualserializedFloat32geometry','partial_image_loader_scope':{'primary_full_tiles':True,'unmeasured_example':[31,29],'active_loader_count_in_bottom_right_block':256-len(missing),'missing_required_first_tile_loader_ids':missing,'diagnosis':'Invalidpixelthreadsreturnbeforecooperativesurfelloads;requiredloaderIDsabsent. Thisismissingdatadependency,notclaimthatearlyreturnaloneviolatesCUDAbarrier. Nopartial-imageGPUfailurewasmeasured.','scope':'Partialimagesnotcertified;frameworkfeaturenotdeclaredunsupported'},'scope':'Positive-depthboundeddiscsfixture,noarbitrarybehind-camera/parallel-ray/opticalcalibrationclaim'}
(b/'study_fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'nonzero_pixels_per_focal':hits,'missing_loaders_unmeasured_partial_case':len(missing)}))
