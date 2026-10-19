"""CPU-only complete saved-array/byte-binding audit; never launches GPU trials."""
import hashlib,json,sys
from pathlib import Path
from audit_native_outputs import audit

def review(result_path):
 result=json.loads(Path(result_path).read_text());assert result['matched_performance_valid']
 checked=[];cache={};means={};case_means={}
 for model in ['cuda','cuda_jl','ka','jacc']:
  d=result['models'][model];assert d['complete']and d['status']=='native_check_pass'and d['timing_valid']
  assert len(d['trials'])==5
  means[model]=sum(t['time_seconds']for t in d['trials'][1:])/4
  assert means[model]==d['mean_seconds']
  case_means[model]=[sum(t['case_times_seconds'][i]for t in d['trials'][1:])/4 for i in range(3)]
  for t in d['trials']:
   assert t['correct']and t['checked_values']==1357921
   a=t['output_artifact'];p=Path(a['saved_path']);digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==a['sha256']and p.stat().st_size==1357921*8
   if digest not in cache:cache[digest]=audit(p)
   checked.append({'model':model,'trial':t['trial'],'path':str(p),'sha256':digest,'all_pass':cache[digest]['all_pass']})
 for model in ['ka','jacc']:
  for reference in ['cuda','cuda_jl']:
   rows=result['comparisons'][model][reference];assert rows['status']=='direct_artifact_pass'and len(rows['trials'])==5
   for t in rows['trials']:assert t['status']=='pass'and t['input_prefix_check']['status']=='pass'and t['input_prefix_check']['values']==1016993
 assert result['reference_agreement']
 return {'all_pass':True,'checked_trials':20,'full_saved_artifacts_rehashed':20,'distinct_full_records_independently_recomputed':len(cache),'independent_audits_by_SHA256':cache,'saved_records':checked,'mean_last_four_seconds':means,'mean_last_four_case_seconds':case_means,'source_of_actual_artifacts':'Preserved independently executed trial outputs; rehash every actual file. Recompute mathematical oracle for each distinct byte-identical full record, not another GPU run.'}
if __name__=='__main__':
 print(json.dumps(review(sys.argv[1]),indent=2))
