"""Full saved-capture review including exact nonfloating semantic fields/guards."""
from pathlib import Path
import struct,json,hashlib,sys,math
p=Path(__file__).resolve().parent;read=lambda b:struct.unpack('<'+'d'*(len(b)//8),b)
fraw=(p/'fixtures_f64.bin').read_bytes();f=read(fraw);oracle=read((p/'oracle_f64.bin').read_bytes());info=json.loads((p/'independent_fixture.json').read_text());rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());checks=[]
for model,data in r['models'].items():
 for t in data.get('trials',[]):
  a=Path(t['output_artifact']['saved_path']);raw=a.read_bytes();v=read(raw);assert len(v)==46840
  assert raw[:len(fraw)]==fraw;pos=len(f);op=0;maxerr=0.;ratio=0.
  for case in info['cases']:
   for i in range(case['lookups']):
    row=v[pos:pos+9];ref=oracle[op:op+9];pos+=9;op+=9;assert row[:5]==ref[:5] and all(map(math.isfinite,row))
    for x,y in zip(row[5:],ref[5:]):
     err=abs(x-y);budget=1e-9+1e-8*abs(y);assert err<=budget;maxerr=max(maxerr,err);ratio=max(ratio,err/budget)
   assert raw[pos*8:(pos+4)*8]==struct.pack('<4d',*([-777.]*4));pos+=4
  assert raw[pos*8:(pos+len(f))*8]==fraw;pos+=len(f)
  assert raw[pos*8:]==struct.pack('<38d',*([-777.]*38));assert pos+38==len(v)
  checks.append({'model':model,'trial':t['trial'],'full_input_and_readonly_fields_exact_bytes':True,'all70_guard_fields_exact_bytes':True,'full_timed_argmax_and_untimed_RNG_fields_exact':True,'full_replayed_macroXS_within_predeclared_budget':True,'max_abs_XS_error':maxerr,'max_normalized_budget':ratio,'sha256':hashlib.sha256(raw).hexdigest()})
print(json.dumps({'result':str(rp),'count':46840,'prefix':len(f),'checked_trials':checks,'all_pass':True,'scope':'Alltimedverificationoutputs+untimedreplaydiagnostics, nophysicalFaddeeva/transportaccuracycertificate; exactguards/readonlynotweakenedbyglobalfloattolerance.'},indent=2))
