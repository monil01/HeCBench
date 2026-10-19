"""Deterministic undirected CSR fixtures and independent BFS minimum labels."""
from pathlib import Path
import struct,json,hashlib,sys,collections
P=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;P.mkdir(parents=True,exist_ok=True)
G=-123456789;cases=[];prefix=[];outputs=[]
for ci,n in enumerate([1,65,4099]):
 adj=[set() for _ in range(n)]
 def edge(a,b):
  if a!=b:adj[a].add(b);adj[b].add(a)
 if n==65:
  for i in range(1,32):edge(i-1,i)
  for i in range(33,64):edge(32,i)
 if n==4099:
  for i in range(2047):edge(i,2047)
  for v in range(512,544):
   for i in range(64):edge(v,i)
  for i in range(2049,4096):edge(i-1,i)
 idx=[0];lst=[]
 for a in adj:lst+=sorted(a,reverse=True);idx.append(len(lst))
 labels=[-1]*n
 for start in range(n):
  if labels[start]!=-1:continue
  q=collections.deque([start]);labels[start]=start
  while q:
   v=q.popleft()
   for w in adj[v]:
    if labels[w]==-1:labels[w]=start;q.append(w)
 A=[G]*16+idx+[G]*16;B=[G]*16+(lst if lst else[G])+[G]*16
 for name,data in [('idx',A),('adj',B),('labels',labels)]:
  (P/f'case{ci}_{name}_i32.bin').write_bytes(struct.pack('<'+'i'*len(data),*data))
 # Actualdevice observer params n,e,blocks; geometry640 is actualV100 source occupancy grid.
 prefix += [n,len(lst),100,640,n,len(lst),640]+A+B+A+B
 outputs += [G]*16+labels+[G]*16+[G]*32
 cases.append(dict(id=ci,n=n,e=len(lst),repeat=100,blocks=640,idx_count=len(A),adj_count=len(B),components=len(set(labels)),degrees=dict(low=sum(len(a)<=16 for a in adj),medium=sum(16<len(a)<=352 for a in adj),high=sum(len(a)>352 for a in adj))))
expected=prefix+outputs;(P/'oracle_i32.bin').write_bytes(struct.pack('<'+'i'*len(expected),*expected))
(P/'study_cases.h').write_text('struct Case{int id,n,e,repeat,blocks,ic,ac;};\nstatic const Case cases[]={'+','.join('{%d,%d,%d,%d,%d,%d,%d}'%(c['id'],c['n'],c['e'],100,640,c['idx_count'],c['adj_count'])for c in cases)+'};\nconstexpr int PREFIX=%d,COUNT=%d,GUARD=%d;\n'%(len(prefix),len(expected),G))
files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in sorted(P.glob('*.bin'))}
(P/'study_plan.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(prefix),artifact_count=len(expected),guard=G,files=files,oracle='Independent Python BFS on undirected sets, ascending root => minimum component ID; not source concurrent union-find.',geometry='Source V10080SM*2048maxthreads/256=640 blocks; adapter asserts actual device geometry equals declared640.',scope='Three synthetic CSR graphs, sourcefiveGPUstages/R100; originalexternaldelaunaydataset absent. Intermediate worklist order nondeterministic, finalminimumlabels exact.'),indent=2)+'\n')
print('CC_FIXTURES',len(prefix),len(expected),cases)
