#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;values=np.fromfile(P/'study_input_f32.bin',dtype='<f4');assert len(values)==770058
pos=0
def read(n):
 global pos
 r=values[pos:pos+n].astype(np.float64);pos+=n;return r
x=read(128*784).reshape(784,128,order='F');weights=[read(i*o).reshape(i,o,order='F')for i,o in [(784,512),(512,512),(512,10)]];biases=[read(o)for o in [512,512,10]];assert pos==len(values)
for ub,ur in [(0,0),(1,0),(0,1),(1,1)]:
 y=x;layers=[]
 for w,bias in zip(weights,biases):
  y=w.T@y
  if ub:y+=bias[:,None]
  if ur:y=np.maximum(y,0)
  layers.append(y.ravel(order='F'))
 expected=np.concatenate(layers).astype('<f8');assert len(expected)==132352 and np.isfinite(expected).all();expected.tofile(P/f'study_expected_b{ub}_r{ur}_f64.bin')
meta={'generator':'SourceC++std::default_random_engine/uniform_float_distribution[-1,1], seed123, exactsource draworder input/weights/biases; generate_study_fixture.cu','oracle':'IndependentFloat64NumPy3layerdense/bias/ReLU; all3intermediates/finaloutputs, no trainedmodelorlabels','numpy':np.__version__,'shape':{'layers':3,'batch':128,'input':784,'hidden':512,'output':10},'input_prefix_count':770058,'output_values':132352,'output_stage_sizes':[65536,65536,1280],'source_seed':123,'files':{q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in P.glob('study_*.bin')}};(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['shape'],meta['files'])
