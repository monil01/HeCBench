from pathlib import Path
import json,struct,math,hashlib,sys
p=Path(__file__).resolve().parent;m=json.loads((p/'fixture_description.json').read_text());load=lambda f:list(struct.unpack('<'+'d'*(Path(f).stat().st_size//8),Path(f).read_bytes()));prefix=load(p/'fixture.bin');gold=load(p/'oracle.bin');rows=[]
for arg in sys.argv[1:]:
 d=json.loads((Path(arg)/'result.json').read_text())
 for model,v in d['models'].items():
  for t in v.get('trials',[]):
   f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert len(raw)==m['artifact_values']*8;assert raw[:len(prefix)*8]==(p/'fixture.bin').read_bytes();a=load(f)[len(prefix):];pos=0;ma=[0.,0.];mr=[0.,0.];norm=[0.,0.];guards=0;probes=0
   for c in m['cases']:
    for k in range(2):
     x=a[pos:pos+272];r=gold[pos:pos+272];pos+=272;assert x[:8]==r[:8] and x[-8:]==r[-8:];guards+=16
     for xx,rr in zip(x[8:-8],r[8:-8]):
      assert math.isfinite(xx) and xx>=0;e=abs(xx-rr)
      if c['deterministic_zero_payoff']:assert xx==rr==0
      else:assert e<=.002+.002*abs(rr);ma[k]=max(ma[k],e);mr[k]=max(mr[k],e/(abs(rr)+1e-300));norm[k]=max(norm[k],e/(.002+.002*abs(rr)))
    x=a[pos:pos+1536];r=gold[pos:pos+1536];pos+=1536;assert struct.pack('<'+'d'*1536,*x)==struct.pack('<'+'d'*1536,*r);probes+=1536
    for n,marker in [(1024,1515870810)]+[(256,-12345.)]*7:
     x=a[pos:pos+n+16];r=gold[pos:pos+n+16];pos+=n+16;assert struct.pack('<'+'d'*len(x),*x)==struct.pack('<'+'d'*len(r),*r);assert x[:8]==[marker]*8 and x[-8:]==[marker]*8;guards+=16
   assert pos==len(gold);assert t['RNG_probe_outside_timing'] and t['paths_per_parameter']==512;rows.append(dict(model=model,trial=t['trial'],all_checks_pass=True,artifact=str(f),sha256=hashlib.sha256(raw).hexdigest(),output_mean_variance_values=2048,exact_RNG_probe_values=probes,exact_input_echo_payload_values=11264,exact_guard_check_values=guards,max_abs_errors=dict(zip(['mean','sample_variance'],ma)),max_rel_errors=dict(zip(['mean','sample_variance'],mr)),max_normalized_budget_errors=dict(zip(['mean','sample_variance'],norm))))
assert rows
print(json.dumps(dict(all_pass=all(x['all_checks_pass'] for x in rows),rows=rows,math_scope=m['mathscope'],rng_probe_scope='Exactfinal4states/first2uniformperlane, nofullcaptureallrandomdraws; probeitselfhasnounguarded-outputclaim: diagnosticallocationnotguarded, measuredmeanvariance/inputbuffersguarded',unique_physical_guard_words=592),indent=2))
