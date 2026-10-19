#!/usr/bin/env python3
"""Deterministic signed-graph fixtures and independent graph-algorithm witnesses."""
from pathlib import Path
import json,struct,hashlib,collections
P=Path(__file__).resolve().parent

def h(x):
 x &= 0xffffffff
 for _ in range(2): x=(((x>>16)^x)*0x45d9f3b)&0xffffffff
 x=(x>>16)^x
 return x if x<0x80000000 else x-0x100000000

def oracle(n,edges,iterations):
 adj=[[]for _ in range(n)];weights={}
 for u,v,w in edges:adj[u].append(v);adj[v].append(u);weights[u,v]=weights[v,u]=int(w==-1)
 for row in adj:row.sort()
 roots=sorted(range(n),key=lambda u:(-len(adj[u]),u))[:iterations]
 assert len({len(adj[r])for r in roots})==iterations
 accnode=[0]*n;acctree=collections.Counter();accneg=collections.Counter();traces=[]
 for it,r in enumerate(roots):
  d=[-1]*n;d[r]=0;q=collections.deque([r])
  while q:
   u=q.popleft()
   for v in adj[u]:
    if d[v]<0:d[v]=d[u]+1;q.append(v)
  parent=[-1]*n;encoded=[0]*n;parent[r]=r;encoded[r]=2147483644
  for v in sorted(range(n),key=lambda v:d[v]):
   if v==r:continue
   bit=((d[v]-1)&1)|2;seed=(it+17)**2+it+17
   u=max((u for u in adj[v]if d[u]+1==d[v]),key=lambda u:(h(((u<<2)|bit)^(v^seed)),(u<<2)|bit))
   parent[v]=u;encoded[v]=(u<<2)|(bit if d[v]==max(d)else 0)
  subtree=[1]*n;potential=[0]*n
  for v in sorted(range(n),key=lambda v:d[v],reverse=True):
   if v!=r:subtree[parent[v]]+=subtree[v]
  for v in sorted(range(n),key=lambda v:d[v]):
   if v!=r:potential[v]=potential[parent[v]]^weights[v,parent[v]]
  minus={};tree={}
  for u in range(n):
   for v in adj[u]:
    t=(parent[v]==u and v!=r)or(parent[u]==v and u!=r);tree[u,v]=int(t)
    minus[u,v]=weights[u,v]if t else(potential[u]^potential[v]if u<v else 1)
    acctree[u,v]+=int(t);accneg[u,v]+=minus[u,v]
  par=list(range(n))
  def find(v):
   while par[v]!=v:par[v]=par[par[v]];v=par[v]
   return v
  for u in range(n):
   for v in adj[u]:
    if not minus[u,v]:
     a,b=find(u),find(v)
     if a!=b:par[max(a,b)]=min(a,b)
  labels=[find(v)for v in range(n)];sizes=collections.Counter(labels);hi=max(sizes,key=lambda x:(sizes[x],x));quot={x:set()for x in sizes};pairs=[]
  for u in range(n):
   for v in adj[u]:
    a,b=labels[u],labels[v]
    if a<b:quot[a].add(b);quot[b].add(a);pairs.append((a,b))
  dist={hi:0};q=collections.deque([hi])
  while q:
   a=q.popleft()
   for b in quot[a]:
    if b not in dist:dist[b]=dist[a]+1;q.append(b)
  count=[dist[v]if v in sizes else 0 for v in range(n)]
  for v in range(n):accnode[v]+=(dist[labels[v]]%2)^1
  traces.append(dict(root=r,depth=max(d),distance=d,parent=parent,encoded_parent=encoded,subtree=subtree,potential=potential,labels=labels,count=count,hi=(sizes[hi]<<32)|hi,pairs=sorted(pairs),minus=[minus[u,v]for u in range(n)for v in adj[u]],tree=[tree[u,v]for u in range(n)for v in adj[u]],inCC=accnode.copy(),inTree=[acctree[u,v]for u in range(n)for v in adj[u]],negCnt=[accneg[u,v]for u in range(n)for v in adj[u]]))
 return roots,traces

cases=[]
for n,it,kind in [(33,1,'positive'),(65,3,'frustrated'),(129,3,'balanced')]:
 pairs={(0,v)for v in range(1,n)}|{(v,v+1)for v in range(1,n-1)}|{(1,v)for v in range(2,n//2+1)}|{(2,v)for v in range(3,n//4+1)}
 edges=[]
 for u,v in sorted(pairs):
  if kind=='positive':w=1
  elif kind=='balanced':w=-1 if ((u*7+u//3)^(v*7+v//3))&1 else (0 if(u+v)%7==0 else 1)
  else:w=(-1,0,1)[(u*17+v*31)%3]
  edges.append((u,v,w))
 roots,traces=oracle(n,edges,it);adj=[[]for _ in range(n)]
 for u,v,w in edges:adj[u].append((v,w));adj[v].append((u,w))
 idx=[0];lst=[];wei=[]
 for row in adj:
  for v,w in sorted(row):lst.append(v);wei.append(w)
  idx.append(len(lst))
 name=f'graph_{n}.csv';(P/name).write_text('source,destination,weight\n'+''.join(f'{100+u*3},{100+v*3},{w}\n'for u,v,w in edges))
 c=dict(n=n,iterations=it,kind=kind,csv=name,nindex=idx,nlist=lst,eweight=wei,origID=[100+v*3 for v in range(n)],roots=roots,traces=traces);cases.append(c)
(P/'fixtures.json').write_text(json.dumps({'cases':cases},indent=2)+'\n')
# Flat binary: case count, then n/e/iterations, CSR, original IDs, roots, expected per-iteration semantic arrays.
a=[len(cases)]
for c in cases:
 n=c['n'];e=len(c['nlist']);a += [n,e,c['iterations']]+c['nindex']+c['nlist']+c['eweight']+c['origID']+c['roots']
 for t in c['traces']:
  a += [t['depth']]+t['encoded_parent']+t['subtree']+t['potential']+t['labels']+t['count']+[t['hi'],len(t['pairs'])]+[v for row in t['pairs']for v in row]+t['minus']+t['tree']+t['inCC']+t['inTree']+t['negCnt']
(P/'fixtures.bin').write_bytes(struct.pack('<'+'q'*len(a),*a))
print(json.dumps({'cases':[(c['n'],len(c['nlist']),c['roots'])for c in cases],'fixture_bytes':8*len(a),'sha256':hashlib.sha256((P/'fixtures.bin').read_bytes()).hexdigest()}))
