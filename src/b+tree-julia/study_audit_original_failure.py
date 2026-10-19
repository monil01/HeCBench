from pathlib import Path
from array import array
import struct,json
import sys
R=Path(__file__).resolve().parents[2];f=(Path(__file__).resolve().parent/'fixtures.bin').read_bytes();pos=4;cs=[]
for ci in range(3):
 n,nn,h,q=struct.unpack_from('<4i',f,pos);pos+=16;parts=[]
 for size in [nn*517,n,q,q,q]:parts.append(list(struct.unpack_from('<'+str(size)+'i',f,pos)));pos+=size*4
 cs.append((n,nn,h,q,parts))
p=Path(sys.argv[1]);v=array('q');v.frombytes(p.read_bytes());pos=0;errors=[]
def ex(x,label):
 global pos
 y=v[pos]
 if x!=y and len(errors)<30:errors.append((pos,label,x,y))
 pos+=1
ex(3,'nc')
for ci,(n,nn,h,q,parts) in enumerate(cs):
 for x in [n,nn,h,q]:ex(x,f'{ci} hdr')
 for ai,part in enumerate(parts):
  for j,x in enumerate(part):ex(x,f'{ci} input{ai} idx{j}')
 for a in range(9):
  for j in range(q):ex(-777 if a==2 or a>=7 else 0,f'{ci} init{a} j{j}')
for ci,(n,nn,h,q,parts) in enumerate(cs):
 nodes,recs,keys,starts,ends=parts
 def leaf(k):
  node=0
  for level in range(h):
   lanes=[i for i in range(256)if nodes[node*517+258+i]<=k<nodes[node*517+258+i+1]]
   assert len(lanes)==1
   node=nodes[node*517+1+lanes[0]]
  return node
 for a in range(2):
  for j,k in enumerate(keys):ex(leaf(k),f'{ci} pointpath{a} j{j} key{k}')
 for j,k in enumerate(keys):
  z=k+2*n;ex(k*7+3 if z>=0 and z%3==0 and z//3<n else -777,f'{ci} pointans j{j} key{k}')
 for a in range(4):
  for j,k in enumerate(starts if a<2 else ends):ex(leaf(k),f'{ci} rangepath{a} j{j} key{k}')
 for j,k in enumerate(starts):ex((k+2*n)//3,f'{ci} rangestart j{j} key{k}')
 for j,(a,b)in enumerate(zip(starts,ends)):ex((b-a)//3+1,f'{ci} rangelen j{j}')
 for j in range(1792):ex(-777,f'{ci} guard{j}')
print(json.dumps({'file':str(p),'count':len(v),'parsed':pos,'mismatches_first30':errors},indent=2))
