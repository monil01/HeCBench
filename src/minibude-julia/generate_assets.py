#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib,argparse
A=argparse.ArgumentParser();A.add_argument("--output-dir",type=Path,required=True);A.add_argument("--asset-dir",type=Path,default=Path(__file__).resolve().parent/"original_assets/data/bm1");args=A.parse_args()
SOURCE=Path(__file__).resolve().parent
assert args.output_dir.resolve()!=SOURCE,"Never overwrite frozen measured fixtures"
args.output_dir.mkdir(parents=True,exist_ok=True)
P=args.output_dir.resolve();D=args.asset_dir.resolve();plan=json.loads((SOURCE/'study_plan.json').read_text());ATOM=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('type','<i4')]);FF=np.dtype([('hb','<i4'),('radius','<f4'),('hphb','<f4'),('elsc','<f4')]);pro=np.fromfile(D/'protein.in',dtype=ATOM);lig=np.fromfile(D/'ligand.in',dtype=ATOM);ff=np.fromfile(D/'forcefield.in',dtype=FF);poses=np.fromfile(D/'poses.in',dtype='<f4').reshape(6,65536);records=[]
for k,case in enumerate(plan['primary_cases']):
 n=case['n']
 if k<3:
  pp=pro.copy();ll=lig.copy();f=ff.copy();ix=np.arange(65536)if k==0 else np.array([0])if k==1 else np.linspace(0,65535,n,dtype=np.int64);t=np.asarray(poses[:,ix],dtype='<f4')
 elif k==3:
  pp=np.zeros(1,dtype=ATOM);ll=np.zeros(1,dtype=ATOM);ll['type']=1;f=np.zeros(34,dtype=FF);f['hb']=70;f['radius']=1;f['hphb'][0]=-3;f['hphb'][1]=-2;f['elsc'][0]=1;f['elsc'][1]=-1;t=np.zeros((6,n),dtype='<f4');t[3]=np.array([0,.5,1,1.5,1.999,2,2.001,2.5,2.999,3,3.001,3.5,3.999,4,4.001,4.5,5,5.5,5.999,6,6.001,6.5,7,7.499,7.5,7.501,8,-1,-2,-3,-7.5],dtype='<f4')
 else:
  pp=np.zeros(8,dtype=ATOM);ll=np.zeros(5,dtype=ATOM)
  for i in range(8):pp[i]=(i*.5-2,(i%3)*.25-.5,(i%4)*.5-1,[0,33,255,256,512,1,127,511][i])
  for i in range(5):ll[i]=(i*.25-.5,(i%2)*.5-.25,(i%3)*.25-.25,[512,256,255,33,0][i])
  f=np.zeros(513,dtype=FF)
  for i in range(513):f[i]=([45,68,69,70][i%4],.75+(i%6)*.125,[-2.,-1.,0.,.5,1.5][i%5],[-1.,-.5,0.,.5,1.][i%5])
  t=np.zeros((6,n),dtype='<f4')
  for j in range(n):t[:,j]=((j%7)*.125,((j*3)%9)*.125,((j*5)%11)*.125,((j%17)-8)*.5,(((j//17)%9)-4)*.5,((j%5)-2)*.5)
  if k==5:
   for name,delta in zip(['x','y','z'],[.5,-1.,2.]):pp[name]+=np.float32(delta)
   t[3:]+=np.array([.5,-1.,2.],dtype='<f4')[:,None]
 assert (f['radius']>0).all()and(pp['type']>=0).all()and(ll['type']>=0).all()and(pp['type']<len(f)).all()and(ll['type']<len(f)).all();row={'index':k,**case,'assets':{}}
 for name,a in [('protein',pp),('ligand',ll),('forcefield',f),('poses',t)]:
  b=np.ascontiguousarray(a).tobytes();path=P/f'case{k}_{name}.bin';path.write_bytes(b);row['assets'][name]={'path':path.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 records.append(row)
prefix=sum(8*(c['protein']+c['ligand']+c['ntypes'])+13*c['n']+1193 for c in records);count=prefix+sum(c['n']+33 for c in records)
(P/'asset_contract.json').write_text(json.dumps({'cases':records,'original_pose_planes_preserved':True,'primary_compiletime_poses_per_thread':1,'source_struct_bytes':16,'source_struct_field_offsets':[0,4,8,12],'input_prefix_count':prefix,'artifact_count':count,'actual_input_prefix_encoding':'RawUInt32words→exactFloat64; allinputbefore+initialoutput+readonlyafter includingactualguardbytes','raw_state_buffers_per_case':10,'actual_allocations_and_source_forcefield_structs_preserved':True},indent=2)+'\n');(P/'study_params.h').write_text('#pragma once\nconstexpr int NCASES=6,R=7;\nconstexpr int NS[6]={'+','.join(str(c['n'])for c in records)+'},NPRO[6]={'+','.join(str(c['protein'])for c in records)+'},NLIG[6]={'+','.join(str(c['ligand'])for c in records)+'},NFT[6]={'+','.join(str(c['ntypes'])for c in records)+'};\n'+f'constexpr int PREFIX={prefix},COUNT={count};\n')
print('24 input assets generated; original BM1 full26x938x65536, controls and513-type stridedsharedloads')
