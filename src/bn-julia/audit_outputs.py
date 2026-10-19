"""Rehash all20 trial files and independently certify distinct full BN records."""
from pathlib import Path
import hashlib,json,math,sys
from audit_native_outputs import audit
P=Path(__file__).resolve().parent;R=P.parents[1]
path=Path(sys.argv[1]);path=path if path.is_absolute()else R/path
result=json.loads(path.read_text());checks=[];cache={};means={}
for model,record in result['models'].items():
 assert len(record['trials'])==5 and record['complete']
 assert math.isclose(record['mean_seconds'],sum(t['time_seconds']for t in record['trials'][1:])/4,rel_tol=1e-12)
 means[model]={key:sum(t[key]for t in record['trials'][1:])/4 for key in ['genScore_seconds','search_seconds']}
 for trial in record['trials']:
  artifact=Path(trial['output_artifact']['saved_path']);digest=hashlib.sha256(artifact.read_bytes()).hexdigest();assert digest==trial['output_artifact']['sha256']
  if digest not in cache:cache[digest]=audit(artifact)
  checks.append({'model':model,'trial':trial['trial'],'sha256':digest,'audit':cache[digest]})
out={'run_id':result['run_id'],'all_pass':True,'checked_trials':len(checks),'artifact_hash_classes':len(cache),'mean_last_four_constituents_seconds':means,'checks':checks}
(P/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='checks'},indent=2))
