#!/usr/bin/env python3
"""Independent E4M3 finite-number-line nearest-even oracle and cache-axis mapping."""
from pathlib import Path
import numpy as np
import json,hashlib
P=Path(__file__).resolve().parent;N,H,D=1024,32,128;Q=N*H*D;scales=np.fromfile(P/'study_scales_f32.bin',dtype='<f4');slots=np.fromfile(P/'study_slot_i64.bin',dtype='<i8');assert np.array_equal(np.sort(slots),np.arange(N));inverse=np.argsort(slots)
values=np.array([(m*2.**-9 if e==0 else (1+m/8.)*2.**(e-7))for e in range(16)for m in range(8)if not(e==15 and m==7)],dtype='f8');assert len(values)==127 and values[-1]==448
codes=[];stats=[]
for tag,scale in zip(['key','value'],scales):
 a=np.fromfile(P/f'study_{tag}_f32.bin',dtype='<f4');assert len(a)==Q and np.isfinite(a).all() and np.isfinite(scale) and scale>0;scaled=np.asarray(a/scale,dtype='f4');mag=np.minimum(abs(scaled.astype('f8')),448.);upper=np.minimum(np.searchsorted(values,mag),126);lower=np.maximum(upper-1,0);dl=mag-values[lower];du=values[upper]-mag;up=(du<dl)|((du==dl)&((upper&1)==0));c=np.where(up,upper,lower).astype('u1')|(np.signbit(scaled).astype('u1')<<7);assert np.all((c&127)!=127);decoded=values[c&127]*np.where((c&128)!=0,-1.,1.)*float(scale);err=decoded-a.astype('f8');stats.append({'input':tag,'scale_float32':float(scale),'max_abs_FP32_to_dequantized_error':float(abs(err).max()),'RMS_FP32_to_dequantized_error':float(np.sqrt(np.mean(err*err))),'encoded_negative_zero_count':int(np.sum(c==128)),'saturation_count':int(np.sum(abs(scaled)>448)),'finite_FP8_codes':True,'nearest_even_ties_count':int(np.sum(dl==du))});codes.append(c.reshape(N,H,D)[inverse])
for B in [16,32]:
 for x in [1,2,4,8]:
  key=np.ascontiguousarray(codes[0].reshape(N//B,B,H,D//x,x).transpose(0,2,3,1,4)).ravel();value=np.ascontiguousarray(codes[1].reshape(N//B,B,H,D).transpose(0,2,3,1)).ravel();np.r_[key,value].tofile(P/f'study_expected_B{B}_x{x}_u8.bin')
(P/'quantization_accuracy.json').write_text(json.dumps({'numpy':np.__version__,'oracle':'Enumerate127finitepositiveE4M3values inDouble; nearest distance/ties-even after roundedFP32division, preservezero sign. Independenttensoraxis transpose basedoninverseslot permutation.','format':'FP8_E4M3finite_saturating_round_to_nearest_even','original_source_FP32_parameters':[N,H,D],'stats':stats,'FP32_application_attention_quality_tested':False},indent=2)+'\n')
(P/'fixture_provenance.json').write_text(json.dumps({'generator_sources':{f:hashlib.sha256((P/f).read_bytes()).hexdigest()for f in ['generate_study_fixture.cpp','generate_study_oracles.py']},'N':N,'heads':H,'dim':D,'seed':42,'source_normal_distribution_draw_order':'key,valueinterleavedfor4Mfeatures, shuffleafteralldraws','oracle_numpy':np.__version__,'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in P.glob('study_*.bin')}},indent=2)+'\n');print(json.dumps(stats))
