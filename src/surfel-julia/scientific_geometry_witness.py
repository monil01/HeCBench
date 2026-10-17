"""Independent surfel geometry and full foreground/background depth witness."""
import pathlib,struct,math,json
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];focals=list(map(f32,[.005,.02,.036]));surfs=[];descriptions=[]
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

import array,sys
result=json.loads(pathlib.Path(sys.argv[1]).read_text());expected=[]
for fi in range(3):expected.extend(maps[fi*1024:(fi+1)*1024]*2)
rows=[]
for model,record in result['models'].items():
 for trial in record['trials']:
  got=array.array('f');got.frombytes(pathlib.Path(trial['output_artifact']['saved_path']).read_bytes());assert len(got)==6144
  error=max(abs(a-b)for a,b in zip(got,expected));assert error<1e-3 and all(math.isfinite(v)for v in got)
  foreground=[]
  for fi in range(3):
   for tiled in range(2):
    image=got[fi*2048+tiled*1024:fi*2048+(tiled+1)*1024];foreground.append(sum(v>0 for v in image))
    assert sum(v>0 for v in image)==hits[fi]
    if fi==0:assert image[25*32+9]==0
    if fi==1:assert image[6*32+17]>0
    if fi==2:assert image[14*32+26]==0 and image[11*32+5]>0
  rows.append({'model':model,'trial':trial['trial'],'all6144_depths_match_Float64_geometry':True,'max_abs_error':error,'foreground_counts_base_tile_each_focal':foreground,'strict_radius_miss_inside_outside_and_tail_witnesses':True})
assert len(rows)==20
print(json.dumps({'benchmark':'surfel','trials':rows,'independent_method':'Float64line-planeprojectionanddiskradiusacceptanceonserializedFloat32geometry;analyticallyaimednear/fardiscsandboundary/tailcases','scope':'32x32fullimageblockspositive-depthgeometry;partialimageloadercoverageonlysourceanalysis,nopartialGPUfailuremeasured'},indent=2))
