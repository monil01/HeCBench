#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,numpy as np,sys
P=Path(__file__).resolve().parent;contract=json.loads((P/'contract.json').read_text());expected=np.fromfile(P/'oracle_full_i32.bin',dtype='<i4');assert len(expected)==contract['count'];assert hashlib.sha256(expected.tobytes()).hexdigest()==contract['oracle_sha256'];rows=[]
for arg in sys.argv[1:]:
 d=json.loads(Path(arg).read_text())
 for model,v in d['models'].items():
  for t in v.get('trials',[]):
   a=t['output_artifact'];f=Path(a['saved_path']);b=f.read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256'];out=np.frombuffer(b,dtype='<i4');assert out.shape==expected.shape and np.array_equal(out,expected)
   off=contract['prefix'];controls=[]
   for k,c in enumerate(contract['cases']):
    n=c['sad_count'];sad=out[off+16:off+16+n].reshape(c['valid_height'],c['valid_width']);off+=n+32;minimum=int(out[off+16]);off+=33;count=int(out[off+16]);off+=33
    assert minimum==min(20,int(sad.min())) and count==int((sad==minimum).sum())
    for y,x in c['known_inserted_matches']:assert sad[y,x]==0
    off+=3*c['width']*c['height']+32;off+=3*c['kernel_width']*c['kernel_height']+32
    controls.append({'kind':c['kind'],'all_valid_window_values':n,'minimum_capped20':minimum,'uncapped_minimum':int(sad.min()),'occurrences_at_capped_min':count,'known_inserted_zero_matches_checked':len(c['known_inserted_matches'])})
   assert off==len(out);rows.append({'model':model,'trial':t['trial'],'checked_values':len(out),'exact_input_prefix':contract['prefix'],'artifact_sha256':a['sha256'],'full_exact_oracle':True,'controls':controls})
(P/'saved_output_audit.json').write_text(json.dumps({'all_pass':True,'full_checked_rows':len(rows),'full_fields_compared_not_scalar_only':True,'rows':rows,'scope':'Independent exact fullvalidwindowL1fixture previouslycomputedfromactualRGBbytes. Hashcheckallactualfiles plusfullvaluecomparison, everyguard/readonlyinput/scalarstate and independentknownmatch/count controls. ThisauditdoesnotrerunfullrealCPUoracle or GPUexecutions.'},indent=2)+'\n');print(json.dumps({'all_pass':True,'full_checked_rows':len(rows),'values_per_trial':len(expected)}))
