#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent
def guarded(a,g):return np.r_[np.full(16,g,dtype='i4'),a.astype('i4').ravel(),np.full(16,g,dtype='i4')]
d=json.loads((P/'study_plan.json').read_text());prefix=[];tail=[]
for k,c in enumerate(d['cases']):
 w,h,kw,kh,n=[c[a] for a in ['width','height','kernel_width','kernel_height','sad_count']];image=np.fromfile(P/f'image{k}.bin',dtype='u1');kernel=np.fromfile(P/f'kernel{k}.bin',dtype='u1');sad=np.fromfile(P/f'oracle{k}.bin',dtype='<i4');assert image.size==3*w*h and kernel.size==3*kw*kh and sad.size==n
 prefix.extend([w,h,kw,kh,n,kw*kh,3]);prefix.extend(guarded(image,231));prefix.extend(guarded(kernel,231));prefix.extend(guarded(np.full(n,-777,dtype='i4'),-99999));prefix.extend(guarded(np.array([20]),-99999));prefix.extend(guarded(np.array([0]),-99999))
 tail.extend(guarded(sad,-99999));tail.extend(guarded(np.array([c['intended_min_capped_at20']]),-99999));tail.extend(guarded(np.array([c['intended_occurrences_at_capped_min']]),-99999));tail.extend(guarded(image,231));tail.extend(guarded(kernel,231))
params='#pragma once\n'
for name,key in [('WS','width'),('HS','height'),('KWS','kernel_width'),('KHS','kernel_height'),('NS','sad_count'),('MINS','intended_min_capped_at20'),('COUNTS','intended_occurrences_at_capped_min'),('OMINS','predicted_original_min_capped20'),('OCOUNTS','predicted_original_occurrences')]:params+='const int '+name+'[5]={'+','.join(str(c[key]) for c in d['cases'])+'};\n'
(P/'study_params.h').write_text(params)
out=np.array(prefix+tail,dtype='<i4');out.tofile(P/'oracle_full_i32.bin');h=params+f'const int PREFIX={len(prefix)},COUNT={len(out)};\n';(P/'study_contract.h').write_text(h);c={'prefix':len(prefix),'count':len(out),'oracle_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'cases':d['cases'],'atol':0,'rtol':0,'source_math_reference_repair':'Fullvalid-windowbounds+compactrowstride (2semanticlines) in compute_sad_array only, exactoriginalGPUwitness first. Min/count original sourceunchanged.','declared_before_target_generation':True,'independent_oracle':'ExactintegerRGBL1 fullvalidwindow, floor normalization independently equalF32sourceoperation at everyactual position; actualrealDVC plus4controls.','timing':d['source_timing']};(P/'contract.json').write_text(json.dumps(c,indent=2)+'\n');print(json.dumps({'prefix':len(prefix),'count':len(out)}))
