from pathlib import Path
import array,struct,random,json,hashlib,math
from fractions import Fraction
p=Path(__file__).resolve().parent;R=p.parents[1]
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def rn_div(a,b):
 exact=Fraction.from_float(a)/Fraction.from_float(b);bits=struct.unpack('<I',struct.pack('<f',a/b))[0];options=[]
 for v in [bits-1,bits,bits+1]:
  if v<0:continue
  x=struct.unpack('<f',struct.pack('<I',v))[0]
  if math.isfinite(x):options.append((abs(Fraction.from_float(x)-exact),v&1,x))
 return min(options)[2]
def hash32(x):
 x=(((x>>16)^x)*0x045d9f3b)&0xffffffff;x=(((x>>16)^x)*0x045d9f3b)&0xffffffff;return(x>>16)^x
def priority(idx):
 n=len(idx)-1;e=idx[-1];avg=rn_div(f32(e),f32(n));scale=f32(126*avg);out=[]
 for v in range(n):
  deg=idx[v+1]-idx[v]
  if deg==0:out.append(254);continue
  x=f32(f32(deg)-f32(hash32(v))*2**-32);den=f32(avg+x);val=int(rn_div(scale,den));out.append(((val+val)|1)&255)
 return out

def make(n,pairs,name):
 adj=[set()for _ in range(n)]
 for a,b in pairs:
  assert 0<=a<n and 0<=b<n and a!=b;adj[a].add(b);adj[b].add(a)
 idx=[0];lst=[]
 for row in adj:lst+=sorted(row);idx.append(len(lst))
 return(name,idx,lst)
rng=random.Random(271);cases=[make(3,[(0,1),(1,2)],'fractional_priority_cast_path3'),make(33,[],'isolates33'),make(65,[(i,j)for i in range(65)for j in range(i)],'clique65_tie_vertex_priority'),make(257,[(i-1,i)for i in range(1,257)],'path257'),make(1031,[(0,i)for i in range(1,1031)],'star1031'),make(4099,[(rng.randrange(i),i)for i in range(1,4099)],'seeded_tree4099')]
with(R/'src/mis-cuda/internet.egr').open('rb')as io:
 header=array.array('i');header.fromfile(io,2);n,e=header;idx=array.array('i');idx.fromfile(io,n+1);lst=array.array('i');lst.fromfile(io,e)
 cases.append(('shipped_internet_unweighted_CSR',list(idx),list(lst)))
fixture=[len(cases)];oracle=[];info=[]
for name,idx,lst in cases:
 n=len(idx)-1;e=len(lst);assert idx[0]==0 and idx[-1]==e and all(idx[i]<=idx[i+1]for i in range(n));assert all(0<=v<n for v in lst)
 arcs={(v,lst[j])for v in range(n)for j in range(idx[v],idx[v+1])};assert all(a!=b and(b,a)in arcs for a,b in arcs)
 prio=priority(idx);selected=[False]*n
 # Independent total-priority sequentialgreedy, not nativeasynchronousspinning translation.
 for v in sorted(range(n),key=lambda v:(prio[v],v),reverse=True):
  selected[v]=not any(selected[lst[j]]for j in range(idx[v],idx[v+1]))
 status=[254 if v else 0 for v in selected]
 assert all(not(selected[a]and selected[b])for a,b in arcs)
 assert all(selected[v]or any(selected[lst[j]]for j in range(idx[v],idx[v+1]))for v in range(n))
 fixture +=[n,e,3]+idx+lst;oracle+=prio+status;info.append({'name':name,'nodes':n,'directed_edges':e,'selected_vertices':sum(selected),'isolates':sum(idx[i]==idx[i+1]for i in range(n)),'max_degree':max(idx[i+1]-idx[i]for i in range(n))})
(p/'fixtures_i32.bin').write_bytes(struct.pack('<'+'i'*len(fixture),*fixture));(p/'oracle_i32.bin').write_bytes(struct.pack('<'+'i'*len(oracle),*oracle));(p/'independent_oracle.json').write_text(json.dumps({'method':'ExactFloat32nearest-evenprioritydivision certifiedwithFractiondistancetoneighborFloat32values, UInt32hashwrap; independenttotalprioritysequentialgreedyallvertices thenseparateindependence/maximality checks, allstatusbits notsetsizeonly.','equivalence_reason':'Unique(prio,vertexid) totalorder; asyncchosenlocalmaxima have no largerunresolvedneighbor, nonadjacenteligiblemaxima commute with descending-prioritygreedy. FiniteactualchecksnotgeneralGPUmemorymodelproof.','cases':info,'input_words':len(fixture),'oracle_words':len(oracle),'dataset_provenance':{'path':'src/mis-cuda/internet.egr','sha256':hashlib.sha256((R/'src/mis-cuda/internet.egr').read_bytes()).hexdigest(),'used':'Exactunweightednindex/nlist CSR; fileedgeweightsignorednativealgorithm andnotpassedtokernel'},'scope':'Fixed24blocks256threads, source3repeats/namedCSRgraphs+actualshippedunweightedgraph, no arbitrarygraph/self-loop/asymmetry/maximumnotmaximalcardinalitycertificate'},indent=2)+'\n');print({'input_words':len(fixture),'oracle_words':len(oracle),'graphs':info})
