#!/usr/bin/env python3
"""Deterministic weighted CSR fixtures, independent arbitrary-integer Dijkstra oracle."""
from pathlib import Path
import heapq,struct,json,hashlib
p=Path(__file__).resolve().parent;INF=1073741823
CASES=[(33,'disconnected_chain'),(64,'uniform_directed_cycle'),(65,'sparse_connected'),(129,'acyclic'),(257,'sparse_connected'),(513,'two_components'),(1025,'sparse_connected')]
raw=bytearray(struct.pack('<I',len(CASES)));rows=[];prefix=0
for n,kind in CASES:
 adj=[{} for _ in range(n)]
 for u in range(n):
  if kind=='uniform_directed_cycle':adj[u][(u+1)%n]=1
  elif kind=='disconnected_chain':
   if u<n//2-1 or n//2<=u<n-1:adj[u][u+1]=1+u%7
  elif kind=='acyclic':
   for jump in (1,7,31):
    if u+jump<n:adj[u][u+jump]=1+(u*17+jump*3)%37
  else:
   lo,hi=(0,n) if kind!='two_components' else ((0,n//2) if u<n//2 else (n//2,n))
   for jump in (1,7,31,127):
    v=lo+(u-lo+jump)%(hi-lo)
    if u!=v:adj[u][v]=1+(u*17+jump*3)%37
 idx=[0];nei=[];weights=[]
 for a in adj:
  for v,w in sorted(a.items()):nei.append(v);weights.append(w)
  idx.append(len(nei))
 upper=((n+63)//64)*64;oracle=[INF]*(upper*upper)
 for start in range(n):
  dist=[None]*n;dist[start]=0;heap=[(0,start)]
  while heap:
   cost,u=heapq.heappop(heap)
   if dist[u]!=cost:continue
   for v,w in adj[u].items():
    cand=cost+w
    if dist[v] is None or cand<dist[v]:dist[v]=cand;heapq.heappush(heap,(cand,v))
  for v,cost in enumerate(dist):
   if cost is not None:oracle[start*upper+v]=cost
 # independently known closed form for uniform directed cycle
 if kind=='uniform_directed_cycle':assert all(oracle[u*upper+v]==(v-u)%n for u in range(n) for v in range(n))
 vals=[n,len(nei),upper,3,*idx,*nei,*weights,*oracle];raw.extend(struct.pack('<'+'i'*len(vals),*vals))
 # Seven guarded buffers: CSR3, adjacency/row/column3, parameter echo1.
 case_words=len(idx)+2*len(nei)+3*upper*upper+4+7*128;prefix+=case_words
 rows.append({'nodes':n,'edges':len(nei),'upper':upper,'repeat':3,'kind':kind,'prefix_words':case_words,'oracle_sha256':hashlib.sha256(struct.pack('<'+'i'*len(oracle),*oracle)).hexdigest(),'finite_offdiagonal_pairs':sum(oracle[u*upper+v]<INF for u in range(n) for v in range(n) if u!=v)})
(p/'fixtures.bin').write_bytes(raw)
x={'benchmark':'floydwarshall2','configuration_id':'ecl_apsp_source64tile_7CSR_cases33_64_65_129_257_513_1025_R3_v1','cases':rows,'inf':INF,'input_prefix_count':prefix,'artifact_count':2*prefix,'dtype':'int32','atol':0,'rtol':0,'guard':1234567890,'guard_words_each_side':64,'alignment':'256byte offset preserves allocationalignment','initial_matrices':'All3 INF before sourceinit kernels, controlled adapter scaffold outside ROI; no sourceGPUbody changes.','scope':'Complete5sourceGPUkernels, 64tile/1024threads and full3repeat driver preserved on named syntheticgraphs; missing CollegeMsg/defaultsource dataset and1000repeats not certified.','oracle':'Independent arbitrary-integer repeated Dijkstra from actualCSR; directedcycle closedform, disconnected/DAG/component/padding controls. No negativecycles/negativeweights; positivepaths bounded belowINF. After3closed-state compute repetitions, complete adjacency/krows expected equal independently computed paddedAPSP; kcols expected equal its independent64x64block-local transpose, not just checksum.','timing':'Sum7case source initialization and compute interval means, eachR3 and wholecontextcompletion. Initialization runs3 then compute runs3 on same matrix without resets; later computes alreadyclosed as source. Allocation/inputreadback/observation/oracle/fixtureIO outside. One fullwarmup then5checkedtrials, dropfirsttime/meanlast4.','source_GPU_changes':[],'raw_output_identity_claim':True,'fixture_sha256':hashlib.sha256(raw).hexdigest(),'fixture_bytes':len(raw)}
(p/'contract.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'prefix':prefix,'artifact':2*prefix,'fixturebytes':len(raw)}))
