#!/usr/bin/env python3
"""Audit immutable powers against independent Gram-LS oracle and signal invariants."""
from pathlib import Path
import struct,json,argparse,hashlib,math
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--output');a=ap.parse_args();ns=1000;nf=100000;d=struct.unpack('<'+'f'*(2*ns+nf+1),(P/'study_input_f32.bin').read_bytes());y=d[ns:2*ns];f=d[2*ns:-1];scale=d[-1];ref=struct.unpack('<'+'d'*nf,(P/'study_expected_f64.bin').read_bytes());bound=.5*scale*sum(v*v for v in y);den=sum(v*v for v in ref);oracle_peak=max(range(nf),key=ref.__getitem__);r=json.load(open(a.result));out={'benchmark':'lombscargle','configuration_id':r['configuration_id'],'result':a.result,'oracle':'IndependentDouble2x2GramLS,norotatedtauformula','actual_scale':scale,'energy_projection_bound':bound,'oracle_peak_angular_frequency':f[oracle_peak],'method_is_not_data_energy_normalized':True,'models':{}}
for model,m in r['models'].items():
 if not m.get('complete'):continue
 rows=[]
 for q in m['trials']:
  art=q['output_artifact'];raw=Path(art['saved_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==art['sha256'];v=struct.unpack('<'+'f'*nf,raw);assert all(math.isfinite(x)and x>=0 for x in v);peak=max(range(nf),key=v.__getitem__);assert abs(f[peak]-1)<.002 and v[peak]<=bound+.1;errors=[abs(x-z)for x,z in zip(v,ref)];assert max(errors)<=.1;rows.append({'trial':q['trial'],'checked_values':nf,'max_abs_error_vs_Double_Gram_LS':max(errors),'relative_output_L2_error_vs_Double_Gram_LS':math.sqrt(sum(x*x for x in errors)/den),'peak_angular_frequency':f[peak],'peak_power':v[peak],'all_nonnegative_finite':True,'projection_bound_pass':True,'artifact_sha256':art['sha256']})
 assert len(rows)==5;out['models'][model]=rows
s=json.dumps(out,indent=2)+'\n'
if a.output:Path(a.output).write_text(s)
else:print(s)
