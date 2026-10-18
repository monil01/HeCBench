"""Seeded fixture and independent stable Float64 objectives; no GPU execution."""
from pathlib import Path
import struct,math,random,json,hashlib
p=Path(__file__).resolve().parent;r=random.Random(20261018);shapes=[(17,1),(33,257),(65,513),(17,1025),(9,4097)];f=[5.];gold=[];records=[]
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def sp(x):return max(x,0.)+math.log1p(math.exp(-abs(x)))
logits=[-80.,80.,-20.,20.,-10.,10.,-2.,2.,-.5,.5,-1e-7,1e-7,-0.,0.,-1.,1.,4.];targets=[0.,.125,.5,.875,1.]
for case,(o,inner) in enumerate(shapes):
 n=(o+1)*inner+17
 if case<4:
  a=[f32(logits[(i*7+case)%len(logits)]) for i in range(n-17)]+[-12345.]*17
  t=[f32(targets[(i*3+case)%len(targets)]) for i in range(n-17)]+[-12345.]*17
 else:
  a=[f32(r.gauss(0,1)) for _ in range(n-17)]+[-12345.]*17
  t=[f32(r.gauss(0,1)+1) for _ in range(n-17)]+[-12345.]*17
 f.extend([float(o),float(inner),float(n)]);f.extend(a);f.extend(t)
 modes=[]
 for mode in range(3):
  out=[]
  for row in range(o):
   terms=[]
   for j in range(inner):
    l=a[row*inner+j];target=t[row*inner+j]
    value=sp(l)-l*target if mode==0 else (2*target-1)*sp(-l) if mode==1 else (1-target)*sp(l)-l*target
    terms.append(value)
   out.append(math.fsum(terms)/inner)
  if case<4 and mode==0:assert min(out)>=0
  gold.extend(out+[-12345.]*17);modes.append({'mode':mode,'minimum':min(out),'maximum':max(out),'negative_rows':sum(x<0 for x in out)})
 records.append({'case':case,'outer':o,'inner':inner,'physical_input_values':n,'target_domain':'softlabels[0,1]' if case<4 else 'Gaussian+1unbounded_source-like','logit_min':min(a[:-17]),'logit_max':max(a[:-17]),'target_min':min(t[:-17]),'target_max':max(t[:-17]),'modes':modes})
fb=struct.pack('<%df'%len(f),*f);gb=struct.pack('<%dd'%len(gold),*gold);(p/'fixture.bin').write_bytes(fb);(p/'expected_outputs.bin').write_bytes(gb)
(p/'fixture_description.json').write_text(json.dumps({'seed':20261018,'prefix_count':len(f),'output_count':len(gold),'cases':records,'path_order':[f'case{c}_mode{m}' for c in range(5) for m in range(3)],'fixture_sha256':hashlib.sha256(fb).hexdigest(),'gold_sha256':hashlib.sha256(gb).hexdigest(),'oracle':'IndependentFloat64stablealgebraic objectives: standardsoftplus(l)-l*t; logD(2*t-1)*softplus(-l); unjoined(1-t)*softplus(l)-l*t. math.fsumrowmean. NotGPUsourceformula reuse.','scope':'Source3modes,5namedboundary/tailshapes,allfulloutputrows/guards, notdefault2048x50176/fulltrainingaccuracy. Native--use_fast_math retained. Softlabelstandardnonnegativity only; auxiliary/unboundedtargets can legitimately yieldnegativeobjectives.'},indent=2)+'\n')
