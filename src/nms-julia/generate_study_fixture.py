from pathlib import Path
import numpy as np,json,hashlib,sys
p=Path(__file__).resolve().parent;n=4096
boxes=np.empty((n,4),dtype='<f4')
for i in range(n):
 g,lane=divmod(i,8);boxes[i]=[(g%32)*64+(lane%3)*2,(g//32)*64+((lane//2)%3)*2,8+(lane%3)*4,(lane//2+1)/8]
(p/'study_input_f32.bin').write_bytes(boxes.tobytes())
a=boxes.astype(np.float64);x,y,w,s=a.T;area=(w+1)**2;bitmap=np.ones((n,n),dtype=np.uint8);margin=1.
for i in range(n):
 iw=np.maximum(0.,np.minimum(x[i]+w[i],x+w)-np.maximum(x[i],x)+1);ih=np.maximum(0.,np.minimum(y[i]+w[i],y+w)-np.maximum(y[i],y)+1);ratio=iw*ih/area;active=s[i]<s
 bitmap[i,active]=((ratio[active]<float(np.float32(.3)))&(w[active]!=0)).astype(np.uint8)
 if active.any():margin=min(margin,float(np.min(np.abs(ratio[active]-float(np.float32(.3))))))
keep=np.all(bitmap!=0,axis=1).astype(np.uint8);assert 0<int(keep.sum())<n and margin>1e-6
(p/'study_expected_u8.bin').write_bytes(bitmap.tobytes()+keep.tobytes());files={}
for name in ['study_input_f32.bin','study_expected_u8.bin']:
 f=p/name;files[name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(p/'fixture_provenance.json').write_text(json.dumps({'generator':'Deterministic512spatialclusters×8positiveintegerwidthsquareboxes; scoreties/fourlevels, exactFloat32dyadic scores. Noactualdetector/image dataset.','oracle':'IndependentFloat64allpairs asymmetricintersection/higher-scoreboxarea threshold, fullbitmap androwAND; strictscoreinequality, sourceFloat32thresholdpromotedexactly.','numpy':np.__version__,'python':sys.version,'detections':n,'kept':int(keep.sum()),'threshold_float32':float(np.float32(.3)),'minimum_active_threshold_margin':margin,'input_prefix_count_bytes':65536,'pairwise_values':n*n,'keep_values':n,'files':files},indent=2)+'\n');print('N',n,'kept',keep.sum(),'thresholdmargin',margin)
