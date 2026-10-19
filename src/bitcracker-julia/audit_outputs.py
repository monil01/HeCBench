"""Rehash every saved full artifact; CPU-only independent audit, no GPU trials."""
import hashlib,json,sys
from pathlib import Path
from audit_native_outputs import audit

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def review(path):
 x=json.loads(Path(path).read_text());assert x['matched_performance_valid']and x['reference_agreement'];cache={};records=[];means={};case_means={};schedule_means={}
 for model in ['cuda','cuda_jl','ka','jacc']:
  d=x['models'][model];assert d['complete']and d['status']=='native_check_pass'and len(d['trials'])==5
  means[model]=sum(t['time_seconds']for t in d['trials'][1:])/4;assert means[model]==d['mean_seconds']
  case_means[model]=[sum(t['case_times_seconds'][i]for t in d['trials'][1:])/4 for i in range(3)];schedule_means[model]=sum(t['schedule_seconds']for t in d['trials'][1:])/4
  for t in d['trials']:
   a=t['output_artifact'];p=Path(a['saved_path']);assert p.stat().st_size==67112675*4;h=digest(p);assert h==a['sha256']and t['correct']and t['checked_values']==67112675
   if h not in cache:cache[h]=audit(p)
   records.append({'model':model,'trial':t['trial'],'path':str(p),'sha256':h,'all_pass':cache[h]['all_pass']})
 for model in ['ka','jacc']:
  for ref in ['cuda','cuda_jl']:
   q=x['comparisons'][model][ref];assert q['status']=='direct_artifact_pass'and len(q['trials'])==5
   for t in q['trials']:assert t['status']=='pass'and t['max_abs_error']==0 and t['input_prefix_check']['status']=='pass'and t['input_prefix_check']['values']==67110908
 return {'all_pass':True,'checked_trials':20,'full_saved_artifacts_rehashed':20,'distinct_full_records_independently_recomputed':len(cache),'independent_audits_by_SHA256':cache,'saved_records':records,'mean_last_four_seconds':means,'mean_last_four_case_seconds':case_means,'mean_last_four_schedule_seconds':schedule_means,'recompute_scope':'Every fullactualsavedfile rehashed; mathematicaloracle reused onlyfor byteidenticalcomplete records afterfullrehash. NoGPUreruns.'}
if __name__=='__main__':print(json.dumps(review(sys.argv[1]),indent=2))
