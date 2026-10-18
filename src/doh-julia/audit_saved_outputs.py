"""CPU-only full filter outputs, exact input/guards and discrete parameter properties."""
from pathlib import Path
import json,struct,math,hashlib,sys
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();prefix=struct.unpack('<2722d',raw);gold=struct.unpack('<11352d',(p/'oracle.bin').read_bytes());meta=json.loads((p/'fixture_description.json').read_text());off=0
for c in meta['cases']:
 h,w,n=c['h'],c['w'],c['pixels'];pixels=prefix[off:off+n];integ=prefix[off+n:off+2*n];off+=2*n
 for r in range(h):
  for col in range(w):assert integ[r*w+col]==sum(pixels[y*w+x] for y in range(r+1) for x in range(col+1))
 assert all(struct.unpack('<f',struct.pack('<f',x))[0]==x for x in pixels+integ)
assert off==2710;rows=[]
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text())
 for model,d in res['models'].items():
  if not d.get('trials'):continue
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   file=Path(t['output_artifact']['saved_path']);a=file.read_bytes();assert a[:len(raw)]==raw;got=struct.unpack('<11352d',a[len(raw):]);go=0;pathmetrics=[]
   for ci,c in enumerate(meta['cases']):
    n=c['pixels'];pairs=[]
    for sigma in meta['sigmas']:
     vals=got[go:go+n+16];ref=gold[go:go+n+16];go+=n+16;assert vals[:8]==ref[:8] and vals[-8:]==ref[-8:];payload=vals[8:-8];truth=ref[8:-8];assert all(math.isfinite(x) for x in payload);frac=max(abs(x-y)/(1e-4+2e-6*abs(y)) for x,y in zip(payload,truth));assert frac<=1
     echo=got[go:go+n+16];echo_ref=gold[go:go+n+16];go+=n+16;assert echo==echo_ref
     assert all(float(struct.unpack('<f',struct.pack('<f',x))[0])==x for x in payload)
     if c['h']==1:assert all(x==0 for x in payload)
     pairs.append(struct.pack('<'+'f'*n,*payload));pathmetrics.append({'case':ci,'sigma':sigma,'payload_count':n,'max_abs_error':max(abs(x-y) for x,y in zip(payload,truth)),'max_acceptance_fraction':frac,'guards_and_actual_inputecho_exact':True})
    assert pairs[2]==pairs[3] #4 and4.25 truncate tosame size12.
   assert go==11352;assert len(t['path_times_seconds'])==16 and abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   for name,sha in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,(model,name)
   rows.append({'model':model,'trial':t['trial'],'artifact':str(file),'sha256':hashlib.sha256(a).hexdigest(),'paths':pathmetrics,'sigma4_and4p25_output_bits_identical':True,'singlerow_outputs_zero':True,'all_declared_dependencies_current':True})
print(json.dumps({'atol':1e-4,'rtol':2e-6,'prefix_bytes':len(raw),'artifact_values':14074,'path_count':16,'actual_output_pixels_pertrial':5420,'actual_inputecho_values_pertrial':5676,'physical_guard_values_pertrial':512,'actual_GPU_dtype':'Float32, losslessFloat64artifactencoding','independent_oracle':'Direct raw pixel sums, not GPU integral corner formula; exact selected inputintegral provenance audited and allfields checked.','unchanged_native_boundary_policy':'Startclippedbeforeend; inclusive integral corners denote(r,r2]×(c,c2], max0 perbox. Not ideal zero padding or continuous Hessian accuracy.','rows':rows},indent=2))
