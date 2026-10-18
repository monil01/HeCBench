"""Independent whole-capture audit; never runs GPU measurements."""
from pathlib import Path
import json,struct,math,hashlib
from generate_fixture import geo
B=Path(__file__).resolve().parent;R=B.parents[1]
runs={'normal':'20261018T041850023146Z_haversine','stress':'20261018T041944310028Z_haversine'};rows=[];issues=[]
for key,run in runs.items():
 rpath=R/'.porting-state/portable_julia/runs'/run/'result.json';r=json.loads(rpath.read_text());raw=(B/f'input_{key}_f64.bin').read_bytes();points=list(struct.iter_unpack('<4d',raw));n=len(points);expected=[geo(*p) for p in points];g=-1234567.25;prefix=struct.pack('<4d',*([g]*4))+raw+struct.pack('<4d',*([g]*4));budget=1e-3 if key=='stress' else 1e-6;captures={}
 for model,m in r['models'].items():
  captures[model]=[]
  for t in m['trials']:
   art=t['output_artifact'];p=Path(art['saved_path']);data=p.read_bytes();sha=hashlib.sha256(data).hexdigest();got=list(struct.unpack('<'+'d'*(len(data)//8),data));out=got[4*(n+2):];errors=[abs(a-b) for a,b in zip(out[1:-1],expected)];ok=len(got)==5*(n+2) and data[:len(prefix)]==prefix and len(out)==n+2 and out[0]==g and out[-1]==g and all(math.isfinite(v) for v in out) and max(errors)<=budget and sha==art['sha256'];captures[model].append(out[1:-1]);row={'fixture':key,'model':model,'trial':t['trial'],'pass':ok,'artifact_sha256':sha,'values':len(got),'max_abs_geometry_error_km':max(errors),'budget_km':budget};rows.append(row)
   if not ok:issues.append(row)
 for model,values in captures.items():
  for i,out in enumerate(values):
   for ref in ['cuda','cuda_jl']:
    if max(abs(a-b) for a,b in zip(out,captures[ref][i]))>budget:issues.append({'fixture':key,'model':model,'reference':ref,'trial':i+1})
summary={'accepted_runs':runs,'all_four_models_five_trials_two_fixtures':len(rows)==40,'oracle':'Independent Float64 3D unit vectors -> cross norm/dot atan2, spherical R6371km','scope':'Computational sphere distance, not ellipsoid or physical Earth accuracy; named4099+99 pairs only','original_antipodal_NaN_hypothesis':'Not observed in99 selected original-native pairs; no native/kernel/domain clamp repair','checks':rows,'issues':issues}
(B/'output_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'checked_artifacts':len(rows),'issues':issues}));assert not issues
