"""Independent bounded-fixture graph scores from original observations."""
import math,pathlib,re,sys
import numpy as np
raw=(pathlib.Path(__file__).parent/'data45_original.h').read_text()
values=np.array(list(map(int,re.findall(r'-?\d+',raw[raw.index('={')+2:raw.rindex('}')]))))
assert values.size==600*45
data=values.reshape(600,45)[:96,:12]
for name in sys.argv[1:]:
 lines=pathlib.Path(name).read_text().splitlines()
 scores=[float(s.split(':')[1]) for s in lines if s.startswith('score:')]
 graphs=[np.array([list(map(int,row.split())) for row in lines[k+1:k+13]]) for k,s in enumerate(lines) if s=='Best Graph:']
 assert len(graphs)==len(scores)==3
 for g,stated in zip(graphs,scores):
  assert g.shape==(12,12) and np.all((g==0)|(g==1)) and not np.diag(g).any()
  expected=0.
  for node in range(12):
   parents=np.flatnonzero(g[node]);assert len(parents)<=4
   groups=np.zeros(96,dtype=np.int64)
   for bit,col in enumerate(parents):groups+=data[:,col]<<bit
   for state in range(1<<len(parents)):
    cells=data[groups==state,node];n0=int(np.count_nonzero(cells==0));n1=len(cells)-n0
    expected+=math.lgamma(n0+1)+math.lgamma(n1+1)-math.lgamma(n0+n1+2)
  assert abs(expected-stated)<=1e-3,(expected,stated)
  print('PASS complete graph12nodes score independent contingency/lgamma; error',abs(expected-stated),'edges',int(g.sum()))
