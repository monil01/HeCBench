#!/usr/bin/env python3
"""Source-grounded N1024 extruded-field oracle, Float64 arithmetic and table topology.
The original source ignores y in f. Counts use identical edge ownership and tables;
coordinate sums use mathematical Float64 accumulation (atomic Float32 GPU tolerance).
"""
from pathlib import Path
import numpy as np,re,hashlib,json
P=Path(__file__).resolve().parent;source=(P.parent/'marchingCubes-cuda/tables.h').read_text()
def section(name):
 s=source[source.index(name):];s=s[s.index('{')+1:s.index('};')];return [int(v,0)for v in re.findall(r'-?0x[0-9a-fA-F]+|-?\d+',s)]
dist=np.array(section('distinctEdgesTable'),dtype=np.uint16);tri=np.array(section('triTable'),dtype=np.int32);edges=np.array(section('edgeIDTable'),dtype=np.uint8).reshape(12,4)
assert len(dist)==256 and len(tri)==4096
iso=float(np.float32(-.9));axis=(np.arange(1025,dtype=np.float64)-512)/512
X=axis[None,:];Z=axis[:,None];field=1-16*X*Z*Z-4*(X*X+2*Z*Z)
mm1=np.empty((16,68,2))
for z in range(16):
 for x in range(68):
  v=field[z*63:z*63+64,x*15:x*15+16];mm1[z,x]=v.min(),v.max()
active1=[x+68*(y+68*z)for z in range(16)for y in range(68)for x in range(68)if mm1[z,x,0]<=iso<=mm1[z,x,1]]
mm2=np.empty((144,340,2))
for z in range(144):
 for x in range(340):
  v=field[z*7:z*7+8,x*3:x*3+4];mm2[z,x]=v.min(),v.max()
active2=[]
for z in range(144):
 for y in range(340):
  for x in range(340):
   parent=x//5+68*(y//5+68*(z//9))
   if mm1[z//9,x//5,0]<=iso<=mm1[z//9,x//5,1] and mm2[z,x,0]<=iso<=mm2[z,x,1]:active2.append(x+340*(y+340*z))
nv=nt=0;coords=np.zeros(4);local_checks=[];invalid_topology=[]
for z in range(144):
 for x in range(340):
  if not(mm1[z//9,x//5,0]<=iso<=mm1[z//9,x//5,1] and mm2[z,x,0]<=iso<=mm2[z,x,1]):continue
  vertex=np.zeros(128,dtype=np.uint16);cases=[];local_v=local_t=0;cs=np.zeros(4)
  for tz in range(8):
   for ty in range(4):
    for tx in range(4):
     xx=x*3+tx;zz=z*7+tz;cc=0
     for bit,(dx,dy,dz)in enumerate([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]):
      if field[zz+dz,xx+dx]<iso:cc|=1<<bit
     eds=(7&(~1 if tx==3 else 7)&(~2 if ty==3 else 7)&(~4 if tz==7 else 7))<<13;de=int(dist[cc])&eds;n=int(dist[cc])&7 if eds==0xe000 else 0
     tid=tx+4*(ty+4*tz);vertex[tid]=local_v|de;cases.append((tid,cc,n));local_v+=de.bit_count();local_t+=n
     v=field[zz,xx]
     if de&0x8000:
      v1=field[zz,xx+1];cs[1]+=(ty-512)/512;cs[2]+=(zz-512)/512;cs[3]+=((xx*(v1-iso)+(xx+1)*(iso-v))/(v1-v)-512)/512
     if de&0x4000:raise AssertionError('extruded field cannot have y edges')
     if de&0x2000:
      v1=field[zz+1,xx];cs[0]+=(xx-512)/512;cs[1]+=(ty-512)/512;cs[3]+=((zz*(v1-iso)+(zz+1)*(iso-v))/(v1-v)-512)/512
  checksum=0
  for tid,cc,n in cases:
   for edge in tri[16*cc:16*cc+3*n]:
    dx,dy,dz,w=map(int,edges[edge]);vi=int(vertex[tid+dx+4*dy+16*dz]);index=(vi>>(16-w)).bit_count()+(vi&0x1fff);
    if not(0<=index<local_v):invalid_topology.append([x,z,tid,cc,n,int(edge),[dx,dy,dz,w],vi,index,local_v,local_t])
    checksum+=index
  nv+=local_v;nt+=local_t;coords+=cs;local_checks.append([x+340*340*z,local_v,local_t,checksum])
counts=[len(active1),len(active2),nv*340,nt*340];assert counts==[8296,240380,4856560,6101640],counts
coords*=340;coords[1]=counts[2]*(-2/512);scores=coords/counts[2]
prefix=np.r_[iso,dist.astype(np.float64),tri.astype(np.float64),edges.ravel().astype(np.float64)]
expected=np.r_[prefix,np.tile(mm1[:,None,:,:],(1,68,1,1)).ravel(),active1,active2,counts,scores].astype('<f8')
for name,array in [('study_expected_f64.bin',expected),('study_minmax2_f64.bin',mm2.astype('<f8')),('study_tables_u16.bin',dist.astype('<u2')),('study_tables_i32.bin',tri.astype('<i4')),('study_edges_u8.bin',edges.ravel()),('study_input_prefix_f64.bin',prefix.astype('<f8'))]:
 (P/name).write_bytes(array.tobytes())
meta={'generator':'generate_study_fixture.py','field':'source x,z,z extruded polynomial; y ignored','N':1024,'iso_float32':iso,'counts':counts,'normalized_partial_coordinate_sums':scores.tolist(),'single_block_topology_witnesses':local_checks,'invalid_local_vertex_index_count_per_yreplica':len(invalid_topology),'invalid_topology_first32':invalid_topology[:32],'input_prefix_count':len(prefix),'output_count':len(expected),'files':{q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in P.glob('study_*.bin')}}
(P/'study_fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items()if k not in ('single_block_topology_witnesses','invalid_topology_first32')},indent=2))
