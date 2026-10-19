#!/usr/bin/env python3
"""Read exact UInt64 raw pointer sidecars; prove every address and dereference."""
from pathlib import Path
import struct,zlib,json,hashlib,sys
P=Path(__file__).resolve().parent
def cases():
 data=(P/'fixtures.bin').read_bytes();pos=4;out=[]
 for _ in range(struct.unpack_from('<I',data)[0]):
  w,h,mode,ni,ne,nz=struct.unpack_from('<6I',data,pos);pos+=24
  inp=struct.unpack_from('<'+'i'*ni,data,pos);pos+=ni*4
  raw=zlib.decompress(data[pos:pos+nz]);pos+=nz;trace=list(struct.unpack('<'+'q'*ne,raw));ti=0;scales=[]
  while ti<len(trace):
   sw,sh=trace[ti:ti+2]
   if sw<24 or sh<24 or sw>w or sh>h:break
   ti+=2;vec=[]
   for i in range(6):n=trace[ti];ti+=1;vec.append(trace[ti:ti+n]);ti+=n
   pix,sums,sqs,offsets,codes,candidates=vec;scales.append((sw,sh,sums,offsets))
   # Remaining final grouped/drawn vectors cannot be mistaken for width,height.
   if ti>=len(trace)or trace[ti] not in range(24,w+1):break
  out.append(scales)
 return out

def audit(path):
 raw=path.read_bytes();v=struct.unpack('<'+'Q'*(len(raw)//8),raw);pos=0;checked=0;records=0
 for scales in cases():
  for mode in (0,1):
   for w,h,sums,offsets in scales:
    actual_mode,aw,ah,n,base,echo=v[pos:pos+6];pos+=6
    assert(actual_mode,aw,ah,n)==(mode,w,h,2913),(path,mode,w,h,'header')
    assert echo==base and base!=0,(path,'device base echo')
    for off in offsets:
     pointer,value=v[pos:pos+2];pos+=2
     if off==-1:assert pointer==0 and value==0,(path,'null')
     else:
      assert 0<=off<w*h and pointer==base+4*off,(path,'address',off,pointer,base)
      assert value==sums[off],(path,'dereference',off,value,sums[off])
     checked+=1
    records+=1
 assert pos==len(v),(path,'unparsed',len(v)-pos)
 return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'dtype':'uint64','records':records,'full_pointer_and_dereference_pairs_checked':checked,'status':'pass','scope':'Process-local exact pointers/base/deviceecho and CPUdereference mapping, not cross-process rawaddressidentity.'}
if __name__=='__main__':
 files=[]
 for arg in sys.argv[1:]:
  p=Path(arg);files.extend(sorted(p.glob('*_u64.bin'))if p.is_dir()else[p])
 rows=[audit(p)for p in files];print(json.dumps({'files':rows,'all_pass':bool(rows)and all(x['status']=='pass'for x in rows)},indent=2))
