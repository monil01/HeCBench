"""Independent Bayesian contingency-count / lgamma oracle for bounded fixture."""
import itertools,math,pathlib,re,sys
import numpy as np
raw=(pathlib.Path(__file__).parent/'data45_original.h').read_text()
values=np.array(list(map(int,re.findall(r'-?\d+',raw[raw.index('={')+2:raw.rindex('}')]))))
assert values.size==600*45
data=values.reshape(600,45)[:96,:12]
result=[]
for node in range(12):
 others=[n for n in range(12) if n!=node]
 for degree in range(5):
  for parents in itertools.combinations(others,degree):
   groups=np.zeros(96,dtype=np.int64)
   for bit,col in enumerate(parents):groups+=data[:,col]<<bit
   score=0.
   for state in range(1<<degree):
    cells=data[groups==state,node];n0=int(np.count_nonzero(cells==0));n1=len(cells)-n0
    score+=math.lgamma(n0+1)+math.lgamma(n1+1)-math.lgamma(n0+n1+2)
   result.append(score)
expected=np.array(result)
for name in sys.argv[1:]:
 actual=np.fromfile(name,np.float32)
 assert actual.size==expected.size==6744 and np.isfinite(actual).all()
 np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-3)
 print('PASS all6744 scores independent contingency/lgamma oracle; maxabs',np.max(np.abs(actual-expected)))
