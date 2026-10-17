from pathlib import Path
import struct,json,collections
p=Path(__file__).resolve().parent
n=603;adj=[set() for _ in range(n)]
def add(a,b):
 adj[a].add(b);adj[b].add(a)
for a in range(40):
 for b in range(a+1,40):add(a,b)
for y in range(17):
 for x in range(17):
  v=40+y*17+x
  if x<16:add(v,v+1)
  if y<16:add(v,v+17)
for v in range(329,585):add(v,v+1)
idx=[0];edges=[]
for a in adj:edges.extend(sorted(a));idx.append(len(edges))
vals=[n,len(edges)]+idx+edges
(p/'fixture_csr.bin').write_bytes(struct.pack('<'+'i'*len(vals),*vals))
(p/'fixture_definition.json').write_text(json.dumps({'nodes':n,'directed_edges':len(edges),'undirected_edges':len(edges)//2,'degree_histogram':dict(collections.Counter(map(len,adj))),'components':['K40vertices0to39','17x17four-neighborgrid40to328','257vertexpath329to585','17isolates586to602'],'clique_lower_bound':40,'proof':'All40cliquevertices pairwiseadjacent; anypropercoloring requires>=40distinctlabels. Doesnotproveglobaloptimalityforarbitrarygraph.','source_scope':'Namedundirectedself-loop-free disconnectedCSR, includesdegree39highworklist anddegree0/1/2/3/4lowpaths andtailnodes beyond512. NativeoriginalECLalgorithmandgenuineJuliarefcoloralgorithms differ, rawcolorvectors preserved/propertychecked.'},indent=2)+'\n')
print(n,len(edges),len(vals))
