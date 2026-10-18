from pathlib import Path
import struct,json,hashlib,math
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;S=R/'.porting-state/portable_julia/voxelization_reference_review';S.mkdir(exist_ok=True);f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];raw=lambda x:struct.unpack('<I',struct.pack('<f',x))[0];half=lambda x:struct.unpack('<H',struct.pack('<e',x))[0];u=lambda x:struct.pack('<I',x);fixtures=[4];suffix=[];cases=[];guard=0xdeadbeef
for ci,(nv,per,invalid)in enumerate([(256,1,3),(256,2,5),(512,10,7),(256,12,1)]):
 pts=[];groups=[]
 for ix in range(nv):
  x=f32(f32(-54)+f32(f32(ix+.5)*f32(.075)));y=f32(f32(-54)+f32(f32(100.5)*f32(.075)));z=f32(f32(-5)+f32(f32(10.5)*f32(.2)))
  idx=math.floor(f32(f32(x-f32(-54))/f32(.075)));idy=math.floor(f32(f32(y-f32(-54))/f32(.075)));idz=math.floor(f32(f32(z-f32(-5))/f32(.2)));assert(idx,idy,idz)==(ix,100,10)
  pointset=[]
  for k in range(per):
   row=[x,y,z,f32((0 if per==12 else k)/8),f32((ix%16+(0 if per==12 else k))/16)];pts.append(row);pointset.append([raw(v)for v in row])
  key=idz*1440*1440+idy*1440+idx;kept=pointset[:min(per,10)];mean=[]
  for f in range(5):
   v=struct.unpack('<f',u(kept[0][f]))[0]
   for row in kept[1:]:v=f32(v+struct.unpack('<f',u(row[f]))[0])
   mean.append(f32(v/f32(len(kept))))
  groups.append({'key':key,'index':[0,idx,idy,idz],'count_pre':per,'count_post':min(per,10),'points':sorted(kept),'mean_bits':[raw(v)for v in mean],'half_bits':[half(v)for v in mean]})
 for j in range(invalid):pts.append([f32(54+j),f32(0),f32(0),f32(j/8),f32(j/16)])
 flat=[raw(v)for row in pts for v in row];fixtures +=[len(pts)]+flat;suffix.append(nv)
 for g in groups:suffix +=[g['key']]+g['index']+[g['count_pre'],g['count_post']]+sum(g['points'],[])+g['mean_bits']+g['half_bits']
 suffix +=[guard]*(9*128);cases.append({'case':ci,'n':len(pts),'valid_voxels':nv,'points_per_voxel':per,'invalid_points':invalid,'groups':groups,'input_bits':flat,'source_feature_grid_fully_active':True,'overflow_input_restricted_to_identical_features':per==12})
(p/'fixtures_u32.bin').write_bytes(b''.join(map(u,fixtures)));(p/'oracle_u32.bin').write_bytes(b''.join(map(u,suffix)));gold=S/'golden_u32.bin';gold.write_bytes(b''.join(map(u,fixtures+suffix)));info={'cases':cases,'input_prefix_count':len(fixtures),'output_count':len(fixtures+suffix),'guard':guard,'golden_path':str(gold.relative_to(R)),'golden_sha256':hashlib.sha256(gold.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256((p/'fixtures_u32.bin').read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256((p/'oracle_u32.bin').read_bytes()).hexdigest(),'contract':'Completekeyedpointmultisets/counts/mean/halfoutputs, exactactualrawmappingandguards; physicalID/slotorder notclaimedidentical.'};(p/'independent_oracle.json').write_text(json.dumps(info,indent=2)+'\n');print(info['input_prefix_count'],info['output_count'])
