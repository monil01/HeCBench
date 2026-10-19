"""Connected cropped tetrahedral controls; no full-domain/aerodynamic claim."""
from pathlib import Path
from collections import deque
import hashlib,json,struct,math,sys
P=Path(__file__).resolve().parent
source=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/imo/mmonil/kitchen/tegra_parser/rodinia_3.1/data/cfd/fvcorr.domn.097K')
f32=lambda x:struct.unpack('<f',struct.pack('<f',float(x)))[0]
with source.open()as f:
 total=int(f.readline());cells=[]
 for i in range(total):
  tokens=f.readline().split();assert len(tokens)==17
  area=f32(tokens[0]);neighbors=[];normals=[]
  for face in range(4):neighbors.append(int(tokens[1+4*face]));normals.append(tuple(f32(x)for x in tokens[2+4*face:5+4*face]))
  assert area>0 and all(math.isfinite(x)for n in normals for x in n)
  cells.append((area,neighbors,normals))
 assert not f.read().strip()
wing=next(i for i,c in enumerate(cells)if 0 in c[1]);selected=[];seen={wing};queue=deque([wing])
while queue and len(selected)<384:
 i=queue.popleft();selected.append(i)
 for neighbor in cells[i][1]:
  j=neighbor-1
  if neighbor>0 and j not in seen:seen.add(j);queue.append(j)
assert len(selected)==384
rows=[]
for case,size in [('cropped192',192),('cropped384',384)]:
 ids=selected[:size];mapping={q:i+1 for i,q in enumerate(ids)};lines=[str(size)];cut=wingfaces=farfaces=interior=0;closure=0.;pairnormal=0.
 for old in ids:
  area,neighbors,normals=cells[old];items=[format(area,'.9g')]
  for q,n in zip(neighbors,normals):
   if q>0:
    if q-1 in mapping:new=mapping[q-1];interior+=1
    else:new=-1;cut+=1;farfaces+=1
   elif q==0:new=0;wingfaces+=1
   else:new=-1;farfaces+=1
   items += [str(new)]+[format(x,'.9g')for x in n]
  ratio=math.sqrt(sum(sum(n[d]for n in normals)**2 for d in range(3)))/sum(math.sqrt(sum(z*z for z in n))for n in normals);closure=max(closure,ratio)
  lines.append(' '.join(items))
 mesh=P/(case+'.mesh');mesh.write_text('\n'.join(lines)+'\n')
 # Full actual index/normal reciprocity review for retained interiorfaces.
 for old in ids:
  for face,q in enumerate(cells[old][1]):
   if q>0 and q-1 in mapping:
    other=q-1;assert old+1 in cells[other][1];back=cells[other][1].index(old+1);a=cells[old][2][face];b=cells[other][2][back]
    e=math.sqrt(sum((u+v)**2 for u,v in zip(a,b)))/max(math.sqrt(sum(u*u for u in a)),1e-30);pairnormal=max(pairnormal,e)
 assert closure<=2e-6 and pairnormal<=2e-6 and wingfaces>0 and farfaces>0 and interior>0
 rows.append({'case':case,'nel':size,'nelr':size,'original_1based_cell_IDs':[i+1 for i in ids],'start_wing_cell_1based':wing+1,'retained_directed_interior_faces':interior,'wing_faces':wingfaces,'farfield_faces':farfaces,'new_farfield_cut_faces':cut,'maximum_relative_normal_closure':closure,'maximum_relative_interior_face_normal_antisymmetry':pairnormal,'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest()})
# Exact tetrahedral area-normal closure, replicated independent freestream cells.
# Rawinputoutwardnormals are negated by originalhostparser before kernel use.
normals=[(-.5,0.,0.),(0.,-.5,0.),(0.,0.,-.5),(.5,.5,.5)]
line=' '.join([format(f32(1/6),'.9g')]+[q for n in normals for q in ['-1']+[format(x,'.9g')for x in n]])
mesh=P/'freestream192.mesh';mesh.write_text('192\n'+(line+'\n')*192)
rows.insert(0,{'case':'freestream192','nel':192,'nelr':192,'geometry':'192 independent exactclosedtetrahedralcells, allfarfield, originalMach1.2 angle0 uniformsourceinitialization','maximum_relative_normal_closure':0.,'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest()})
(P/'mesh_generation_review.json').write_text(json.dumps({'original_local_public_Rodinia_source':str(source),'original_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'original_cells':total,'native_DVC_default_asset_available_here':False,'source_selection':'LocalRodinia3.1data, notSPECcopy; no assertion thismatchesunfetchedDVCarchive bytes.','physical_scope':'Two connectedcroppedsubdomains, retainedgeometry+wallfaces; cutsnewfarfield. Notfulloriginal97046cellwingdomain, gridconvergence/lift/drag orvalidatedaerodynamicmodel. Third analyticfreestreamcontrol exactclosure.','iterations':2000,'source_stage_weights':'Three stages j0/1/2 use1/(4-j), notclaimedclassicalthirdorderRK.','untriggered_original_host_padding_bug':'Allthree sizesmultiples192, so brokenpaddednormalassignment loopnotentered. Nonmultiple193CLIremainsuncertified, noGPUalgorithmcorrection.','meshes':rows},indent=2)+'\n')
print(json.dumps({'meshes':[{k:v for k,v in r.items()if k!='original_1based_cell_IDs'}for r in rows]},indent=2))
