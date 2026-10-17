"""Independent pair-placement + crossing-graph bipartiteness; no source DFS/hash/stacks."""
from pathlib import Path
import json,struct,time,collections
P=Path(__file__).resolve().parent

def enumerate_planar(n):
 positions=[-1]*n;raw=[];nodes=0
 def find(par,x):
  parity=0
  while par[x][0]!=x:
   nxt,p=par[x];parity^=p;x=nxt
  return x,parity
 def visit(m,occupied,par):
  nonlocal nodes
  nodes+=1
  if m==0:
   roots={find(par,j)[0]for j in range(n)};multiplicity=1<<(len(roots)-1)
   raw.extend([tuple(positions)]*multiplicity);return
  for start in range(2*n-m-1):
   close=start+m+1
   if m==1 and close>n:continue
   bits=(1<<start)|(1<<close)
   if occupied&bits:continue
   newpar=list(par);valid=True
   for j in range(m,n):
    otherclose=positions[j];otherstart=otherclose-j-2
    if (start<otherstart<close<otherclose)or(otherstart<start<otherclose<close):
     a,pa=find(newpar,m-1);b,pb=find(newpar,j)
     if a==b:
      if pa^pb!=1:valid=False;break
     else:newpar[a]=(b,pa^pb^1)
   if valid:
    positions[m-1]=close;visit(m-1,occupied|bits,newpar)
 visit(n,0,[(i,0)for i in range(n)])
 return sorted(raw),nodes

if __name__=='__main__':
 flat=[];meta=[]
 for n in [7,8,11,12,15]:
  t=time.monotonic();raw,nodes=enumerate_planar(n);unique=len(set(raw));assert unique=={7:0,8:4,11:16,12:40,15:194}[n]
  # Fixed artifactcapacity4096 tuples/case, ndefinedbytes only, zero-unusedslots.
  assert len(raw)<4096;flat.extend([n,len(raw),unique]);flat.extend(x for row in raw for x in row);flat.extend([0]*((4096-len(raw))*n));meta.append({'n':n,'raw_embedding_count':len(raw),'unique_position_tuples':unique,'independent_search_nodes':nodes,'seconds':time.monotonic()-t,'oracle_algorithm':'Pair-placementbacktracking,incrementalparityunionfindofcrossinggraph; eachconnectedcomponentcanflipabove/below, fixingfirstpairbelowgives2^(components-1)embeddings; restrictm1closing<=n forreversal.'})
  print(meta[-1],flush=True)
 (P/'expected_i64.bin').write_bytes(struct.pack('<%dq'%len(flat),*flat));(P/'independent_oracle.json').write_text(json.dumps({'cases':meta,'artifact_count':len(flat),'capacity':4096,'scope':'SourceN7,8,11,12,15; largerunmeasured, nopublishedcountaloneusedasoracle'},indent=2)+'\n')
