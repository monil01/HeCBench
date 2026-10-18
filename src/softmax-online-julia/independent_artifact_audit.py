from pathlib import Path
import json,hashlib,struct,math,sys
p=Path('src/softmax-online-julia');run='20261018T012314534260Z_softmax-online';d=json.loads((Path('.porting-state/portable_julia/runs')/run/'result.json').read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();gold=struct.unpack('<'+str(meta['output_count'])+'d',(p/'expected_outputs.bin').read_bytes());assert len(prefix)==627744
checks=[];timings={}
for model,v in d['models'].items():
 ts=v['trials'];assert len(ts)==5 and all(t['correct'] for t in ts);assert math.isclose(v['mean_seconds'],sum(t['time_seconds'] for t in ts[1:])/4,rel_tol=1e-14)
 timings[model]={'all_five_path_times_seconds':[t['per_path_times_seconds'] for t in ts],'mean_path_times_last4_seconds':[sum(t['per_path_times_seconds'][i] for t in ts[1:])/4 for i in range(30)],'mean_sum_seconds':v['mean_seconds']}
 for t in ts:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;got=struct.unpack('<'+str(meta['output_count'])+'f',raw[len(prefix):]);assert len(raw)==1098456*4
  off=0;ma=mass=0.;guards=0
  for case in meta['cases']:
   n,c=case['rows'],case['cols'];size=n*c+17
   for w in case['threads']:
    y=got[off:off+size];g=gold[off:off+size];off+=size
    for x,r in zip(y,g):
     assert math.isfinite(x);e=abs(x-r);ma=max(ma,e);assert e<=1e-4+2**-23*abs(r)
    assert y[n*c:]==(-12345.,)*17;guards+=17
    for row in range(n):
     vals=y[row*c:(row+1)*c];assert all(0<=x<=1+1e-6 for x in vals);e=abs(math.fsum(vals)-1);mass=max(mass,e);assert e<=1e-5
    for a,b in case['shift_pairs_zero_based']:assert struct.pack('<'+str(c)+'f',*y[a*c:(a+1)*c])==struct.pack('<'+str(c)+'f',*y[b*c:(b+1)*c])
  assert off==len(got) and guards==510
  checks.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'max_abs_Float64_error':ma,'max_row_mass_error':mass,'prefix_exact':True,'all_510_guards_exact':True,'shift_pairs_bit_exact':True})
rec={'benchmark':'softmax-online','run':run,'artifacts_checked':20,'input_prefix_count':156936,'input_prefix_bytes':len(prefix),'output_count':941520,'probability_count':941010,'guards_per_artifact':510,'all_artifacts_bit_identical':len(set(x['sha256'] for x in checks))==1,'checks':checks,'per_model_timings':timings,'scope':'Full rawprobabilities/all30paths, exactserializedinputprefix/guards; independentFloat64oracle/unitmass/nonnegativity/exactpairedshiftinvariance. Finiteinputs only; kernel4 notnative1..3/fullCLI.'};(p/'independent_artifact_audit.json').write_text(json.dumps(rec,indent=2)+'\n');print({k:rec[k] for k in ['artifacts_checked','all_artifacts_bit_identical']})
