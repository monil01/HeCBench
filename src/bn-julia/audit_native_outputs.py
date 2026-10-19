"""Independent CPU audit of full native/Julia BN records; no source helper calls.
Use: python3 audit_native_outputs.py ARTIFACT.bin
"""
from pathlib import Path
import array,struct,itertools,math,json,sys,hashlib
P=Path(__file__).resolve().parent

def load(p):
 a=array.array('d');a.frombytes(Path(p).read_bytes());return a
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def floatsum(xs):
 a=0.
 for x in xs:a=f32(a+x)
 return a

def audit(path):
 artifact=load(path);fixture=load(P/'fixture.bin');sourcegold=load(P/'score_oracle.bin');mathgold=load(P/'math_score_oracle.bin');bound=load(P/'math_score_budget.bin');pos=1;fo=1;cases=[];assert artifact[0]==fixture[0]==3
 def take(n):
  nonlocal pos
  v=list(artifact[pos:pos+n]);pos+=n;assert len(v)==n;return v
 for ci in range(3):
  head=take(4);assert head==list(fixture[fo:fo+4]);fo+=4;N,D,states,S=map(int,head);raw=list(fixture[fo:fo+N*D]);fo+=N*D;lg=list(fixture[fo:fo+D+2]);fo+=D+2
  expected=[[ -37.]*32+v+[-37.]*32 for v in [raw,lg,[0.]*(N*S),[0.]*N,[0.],[0.]*4]];expected[3]=[173.]*32+[0.]*N+[173.]*32
  for v in expected:assert take(len(v))==v
  masks=[[take(N+64)for node in range(N)]for step in range(100)]
  for step in masks:
   for m in step:assert m[:32]==[173.]*32 and m[-32:]==[173.]*32 and all(x in (0.,1.)for x in m[32:-32])
  cases.append((N,D,S,expected,masks))
 assert pos==220714 and fo==len(fixture)
 percase=[];co=0;checks=0;maxmath=0.;maxsource=0.;certified_graphs=0
 for ci,(N,D,S,expected,masks)in enumerate(cases):
  local=take(N*S);allcomb=[];ranks=[]
  for node in range(N):
   comb=[c for k in range(5)for c in itertools.combinations([n for n in range(N)if n!=node],k)];assert len(comb)==S;allcomb.append(comb);ranks.append({c:i for i,c in enumerate(comb)})
  for i,x in enumerate(local):
   e=abs(x-mathgold[co+i]);assert math.isfinite(x)and e<=bound[co+i];maxmath=max(maxmath,e);maxsource=max(maxsource,abs(x-sourcegold[co+i]));assert abs(x-sourcegold[co+i])<=1e-3+2e-6*abs(sourcegold[co+i])
  co+=N*S
  def graph_score(graph):
   terms=[]
   for node in range(N):
    par=tuple(k for k,x in enumerate(graph[node*N:node*N+N])if x)
    assert node not in par and len(par)<=4 and par in ranks[node]
    terms.append(local[node*S+ranks[node][par]])
   # Independent Kahn topological test; graph rows contain parents.
   remaining=set(range(N));completed=set()
   while remaining:
    ready={n for n in remaining if all(not graph[n*N+p]or p in completed for p in range(N))};assert ready,'directedcycle';completed|=ready;remaining-=ready
   return floatsum(terms)
  prev=None;populated=[];lastnodes=None
  for step in range(100):
   choices=[];scores=[];rawnodes=[]
   for node in range(N):
    row=take(5);value,encoded=row[0],list(map(int,row[1:]));rawnodes.append(row);par=[]
    for code in encoded:
     if code<=0:break
     actual=code-1 if code<=node else code;assert 0<=actual<N and actual!=node;par.append(actual)
    par=tuple(par);assert tuple(sorted(set(par)))==par and par in ranks[node]
    mask=masks[step][node][32:-32];assert not mask[node];assert all(mask[x]==1 for x in par)
    allowed=[c for c in allcomb[node]if all(mask[x]for x in c)];maximum=max(local[node*S+ranks[node][c]]for c in allowed)
    assert value==maximum and value==local[node*S+ranks[node][par]],('notorderedargmax',ci,step,node,value,maximum,par,local[node*S+ranks[node][par]],[(c,local[node*S+ranks[node][c]])for c in allowed if local[node*S+ranks[node][c]]==maximum],encoded)
    choices.append(par);scores.append(value);checks+=1
   snap=take(573);score,previous_score,count=snap[:3];orders=snap[3:84];graph=snap[84:165];preorders=snap[165:246];pregraph=snap[246:327];topscore=snap[327:330];topgraph=snap[330:]
   assert orders==[x for m in masks[step]for x in m[32:-32]]
   assert all(x in(0.,1.)for x in orders+graph+preorders+pregraph+topgraph)
   for i in range(N):
    assert not orders[i*N+i]
    for j in range(N):
     if i!=j:assert orders[i*N+j]+orders[j*N+i]==1
     for k in range(N):
      if orders[i*N+j]and orders[j*N+k]:assert orders[i*N+k]
   expectedgraph=[float(p in choices[node])for node in range(N)for p in range(N)];assert graph==expectedgraph
   assert score==floatsum(scores)==graph_score(graph);certified_graphs+=1
   if prev is None:assert previous_score==score and pregraph==graph and preorders==orders
   else:assert (previous_score==score and pregraph==graph and preorders==orders)or(previous_score==prev[0]and pregraph==prev[1]and preorders==prev[2])
   assert graph_score(pregraph)==previous_score
   for slot in range(min(int(count),3)):
    g=topgraph[slot*81:(slot+1)*81];assert graph_score(g)==topscore[slot]
   prev=(previous_score,pregraph,preorders);lastnodes=rawnodes;populated.append(min(int(count),3))
  percase.append({'case':ci,'local_scores':N*S,'ordered_argmax_nodes':900,'search_iteration_graphs':100,'final_populated_top_slots':populated[-1],'last_node_results':lastnodes,'last_mask':masks[-1][-1],'local':local})
 for ci,(N,D,S,expected,masks)in enumerate(cases):
  final=[take(len(v))for v in expected]
  for v,e in zip(final,expected):assert v[:32]==e[:32]and v[-32:]==e[-32:]
  assert final[0]==expected[0]and final[1]==expected[1]and final[2][32:-32]==percase[ci]['local']
  assert final[3]==percase[ci]['last_mask'];last=percase[ci]['last_node_results'][-1];assert final[4][32:-32]==last[:1]and final[5][32:-32]==last[1:]
 assert pos==len(artifact)==434116 and co==len(sourcegold)
 for d in percase:
  for name in ['last_node_results','last_mask','local']:d.pop(name)
 return {'all_pass':True,'artifact_words':len(artifact),'input_prefix_exact_words':220714,'independent_local_scores':co,'independent_ordered_argmax_nodes':checks,'independent_DAG_iteration_graphs':certified_graphs,'max_abs_vs_Float64_K2_math':maxmath,'max_abs_vs_source_CPU':maxsource,'cases':percase,'scope':'Withinorder4parentlocaloptima andDAGscorecertificates only;100orderheuristic notglobalDAGoptimality. Unpopulatedtopslots retainedrawbutnotcertifiedasscoredgraphs. Allactualinputs/masks/guards/fullrecordschecked.'}
if __name__=='__main__':
 p=Path(sys.argv[1]);out=audit(p);out['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();print(json.dumps(out,indent=2))
