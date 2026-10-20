#!/usr/bin/env python3
"""Independent coordinate/normal checks of source GPU surface and MPI gather maps."""
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;F=P/'fixtures';meshes=[]
for r in [0,1]:
 xyz=np.stack([np.fromfile(F/f'rank{r}_{a}.bin',dtype='<f8').reshape(511,84)for a in ['x','y','z']],axis=2);surf=np.fromfile(F/f'rank{r}_surfinfo.bin',dtype='<f4').astype(np.float64).reshape(511,7,112);mapping=np.fromfile(F/f'rank{r}_parmapOUT.bin',dtype='<i4');meshes.append((xyz,surf,mapping))
rows=[]
for r,(xyz,surf,mapping)in enumerate(meshes):
 peerxyz,_,peermap=meshes[1-r];M=surf[:,0].astype(np.int64);Pidx=surf[:,1].astype(np.int64);mask=Pidx<0;j=-1-Pidx[mask];sender=peermap[6*j];rc=M[mask]//576;rn=M[mask]%576;pc=sender//576;pn=sender%576;assert np.all((rn<84)&(pn<84));delta=abs(xyz[rc,rn]-peerxyz[pc,pn]);assert np.all(delta<=1e-4),'sourceNODETOL';norm=surf[:,4:7];normerr=abs(np.sum(norm*norm,axis=1)-1);normbudget=(8*2**-24)/(1-8*2**-24)+1e-12;assert np.all(normerr<=normbudget);assert np.all(surf[:,2]>0);assert set(np.unique(surf[:,3]))=={-1.,1.};assert np.count_nonzero(mask)==76*28
 for field in range(6):
  idx=mapping[field::6];assert np.all((idx%576)//96==field);assert np.array_equal(idx-field*96,mapping[::6])
 rows.append(dict(rank=r,remote_face_nodes=int(mask.sum()),cut_faces=76,max_remote_coordinate_abs_difference=float(delta.max()),coordinate_tolerance=1e-4,coordinate_tolerance_source='src/miniDGS-cuda/include/Mesh.h NODETOL',max_unit_normal_squared_error=float(normerr.max()),normal_squared_budget=normbudget,positive_surface_scales=True,gather_fields_actual_global_cartesian_order=True,all_passed=True))
(P/'mesh_topology_witness.json').write_text(json.dumps(dict(CPU_only=True,checks=rows,independent_physical_coordinate_association=True,not_application_trajectory_or_CFL_certificate=True,source_assets=[dict(path=str(p.relative_to(P)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in sorted(F.glob('*.bin'))if any(t in p.name for t in ['_x.','_y.','_z.','_surfinfo.','_parmapOUT.'])]),indent=2)+'\n');print('TOPOLOGY',sum(z['remote_face_nodes']for z in rows),'crossranknodes passed')
