#!/usr/bin/env python3
"""CPU-only DIMACS parsing and independent heap-Dijkstra full-distance oracle."""
import argparse,pathlib,heapq,json,hashlib,struct,collections
INF=2147483647
P=pathlib.Path(__file__).resolve().parent
def generate(name,n,edge_lists,source,out,expected=None):
 starts=[];edges=[]
 for row in edge_lists:starts.extend((len(edges)//2,len(row)));edges.extend(z for v,w in row for z in (v,w))
 e=len(edges)//2;assert all(0<=edges[2*j]<n and edges[2*j+1]>0 for j in range(e));dist=[INF]*n;dist[source]=0;heap=[(0,source)];pops=0
 while heap:
  d,u=heapq.heappop(heap)
  if d!=dist[u]:continue
  pops+=1
  for j in range(starts[2*u],sum(starts[2*u:2*u+2])):
   v,w=edges[2*j:2*j+2];cand=d+w
   if cand<dist[v]:dist[v]=cand;heapq.heappush(heap,(cand,v))
 reachable=[x for x in dist if x!=INF];assert max(reachable,default=0)+max(edges[1::2],default=0)<INF
 tight=[False]*n;tight[source]=True
 for u in range(n):
  for j in range(starts[2*u],sum(starts[2*u:2*u+2])):
   v,w=edges[2*j:2*j+2]
   if dist[u]!=INF:
    assert dist[v]<=dist[u]+w
    if dist[v]==dist[u]+w:tight[v]=True
 assert all(tight[i]or dist[i]==INF for i in range(n))
 if expected:
  a=list(map(int,pathlib.Path(expected).read_text().split()));assert a[0]==n and a[1::2]==list(range(n)) and a[2::2]==dist
 raw=[n,e,source]+starts+[z if i%2==0 else-z for i,z in enumerate(edges)]
 (out/(name+'_graph_i32.bin')).write_bytes(struct.pack('<'+'i'*len(raw),*raw));cost=[-x for x in dist];(out/(name+'_oracle_i32.bin')).write_bytes(struct.pack('<'+'i'*n,*cost));return dict(name=name,n=n,e=e,source=source,reachable=pops,max_distance=max(reachable),max_degree=max(starts[1::2]),max_edge_weight=max(edges[1::2],default=0),independent_oracle='heapDijkstra_positiveweights/fullBellmaninequalities/tightpredecessorcertificate',source_expected_file_agreement=bool(expected),geometry_queue_overflow_scope='Original1600sharedqueuepreserved; actualoverflowflagandfullrawfrontiermembershipchecked, noassumeduniversalmaxdegree8safety.')
def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset-root',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--proof',type=pathlib.Path);z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True);v=list(map(int,(z.dataset_root/'input/NYR_input.dat').read_text().split()));n,e,s=v[:3];ns=v[3:3+2*n];es=v[3+2*n:];assert len(es)==2*e;adj=[[(es[2*j],es[2*j+1])for j in range(ns[2*i],sum(ns[2*i:2*i+2]))]for i in range(n)];rows=[generate('NYR',n,adj,s,z.output,z.dataset_root/'output/NYR_bfs.out')]
 # Positive weighted directed graph: competing long/short paths, cycles, unreachable nodes and deep frontier.
 n=1027;adj=[[]for _ in range(n)]
 for u in range(1000):
  if u+1<1000:adj[u].append((u+1,7+(u%5)))
  if u+17<1000:adj[u].append((u+17,10+(u%3)))
  if u>=2:adj[u].append((u-2,4))
 rows.append(generate('weighted_control',n,adj,0,z.output));n=33;adj=[[]for _ in range(n)]
 for u in range(31):adj[u].append((u+1,1+u%7))
 rows.append(generate('narrow_chain',n,adj,0,z.output));proof=dict(cases=rows,files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in z.output.glob('*_i32.bin')},CPU_only=True)
 if z.proof:z.proof.write_text(json.dumps(proof,indent=2)+'\n')
 print(json.dumps(rows))
if __name__=='__main__':main()
