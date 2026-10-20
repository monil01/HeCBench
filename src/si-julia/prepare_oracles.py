#!/usr/bin/env python3
from pathlib import Path
import argparse,struct,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();manifest=json.loads((a.root/'fixture_manifest.json').read_text());records=[];guard=np.float32(-1234567)
for case in manifest['cases']:
 raw=(a.root/case['file']).read_bytes();k,u,total=struct.unpack_from('<IIQ',raw);sizes=struct.unpack_from('<'+'I'*k,raw,16);elements=struct.unpack_from('<'+'I'*total,raw,16+4*k);ss=[];off=0
 for n in sizes:ss.append(set(elements[off:off+n]));off+=n
 U=u+1;P=case['partition'];tiles=[range(i,min(i+P,k))for i in range(0,k,P)];prefix=[];out=[]
 for i,A in enumerate(tiles):
  aa=np.zeros((P,U),dtype='<f4')
  for ia,index in enumerate(A):aa[ia,list(ss[index])]=1
  prefix.extend([aa.reshape(-1),np.full(17,guard,dtype='<f4')])
  for B in tiles[i:]:
   bb=np.zeros((U,P),dtype='<f4')
   for ib,index in enumerate(B):bb[list(ss[index]),ib]=1
   prefix.extend([bb.reshape(-1),np.full(17,guard,dtype='<f4')]);cc=np.zeros((P,P),dtype='<f4')
   for ia,x in enumerate(A):
    for ib,y in enumerate(B):cc[ia,ib]=len(ss[x]&ss[y])
   out.extend([cc.reshape(-1),np.full(17,guard,dtype='<f4')])
 prefix=np.concatenate(prefix);out=np.concatenate(out);f=a.root/(case['name']+'_oracle_f32.bin');f.write_bytes(np.concatenate([prefix,out]).astype('<f4').tobytes());records.append({'name':case['name'],'path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'input_prefix_count':len(prefix),'output_count':len(out),'total_count':len(prefix)+len(out),'independent_oracle':'Pythonsetsintersections fullallpairsincldiagonals; matricespackedindependentlyoutsideROI, notGPUexpectedoutputcopy','source_set_counts':sizes[:24]})
(a.root/'oracle_manifest.json').write_text(json.dumps({'benchmark':'si','NumPy':np.__version__,'Python_struct_ABI':'<IIQ_header_thenUInt32sizes/elements','guard_value':float(guard),'cases':records},indent=2)+'\n');print([(r['name'],r['bytes'])for r in records])
