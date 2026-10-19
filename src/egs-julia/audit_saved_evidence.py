"""Read-only EGS receipt audit; no GPU execution or measurement selection."""
from pathlib import Path
import argparse,json,hashlib,math
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=json.loads(a.result.read_text());assert D['matched_performance_valid'] and all(D['variant_verified'].values()) and D['reference_agreement'];hashes=[];bindings={};means={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
for model,r in D['models'].items():
 assert r['complete'] and r['timing_valid'] and r['status']=='native_check_pass' and r['provenance_valid'];assert len(r['trials'])==5
 for trial,row in enumerate(r['trials'],1):
  assert row['trial']==trial and row['correct'] and row['jit_excluded'];assert row['actual_histories']==43728 and row['inner_iterations_by_phase']==[32768,64,1];art=row['output_artifact'];path=Path(art['saved_path']);assert art['count']==21646824 and path.stat().st_size==art['count']*8;actual=sha(path);assert actual==art['sha256'];hashes.append({'model':model,'trial':trial,'path':str(path),'sha256':actual,'bytes':path.stat().st_size})
 mean=math.fsum(x['time_seconds']for x in r['trials'][1:])/4;assert mean==r['mean_seconds'];means[model]=mean
 for path,expected in r['adapter_file_hashes'].items():
  q=Path(path);actual=sha(q);assert actual==expected;bindings[path]=actual
for model in ['ka','jacc']:
 for ref in ['cuda','cuda_jl']:
  c=D['comparisons'][model][ref];assert c['status']=='direct_artifact_pass' and len(c['trials'])==5;assert all(t['status']=='pass' and t['checked_values']==21646824 and t['input_prefix_check']['status']=='pass' and t['input_prefix_check']['values']==6438653 for t in c['trials'])
assert len(D['reference_checks'])==5 and all(t['status']=='pass'for t in D['reference_checks'])
result={'all_pass':True,'result_path':str(a.result),'result_sha256':sha(a.result),'five_trial_records':20,'artifact_values_each':21646824,'actual_input_prefix_UInt32_words':6438653,'full_artifact_hash_checks':hashes,'declared_dependency_bindings_checked':bindings,'distinct_dependency_count':len(bindings),'means_last_four_seconds':means,'native_GPU_source_byte_identical':sha(Path(__file__).resolve().parent/'configured_native/kernels.cu')==sha(Path(__file__).resolve().parent.parent/'egs-cuda/kernels.cu'),'acceptance':'Allindependentper-trialgates + fivefullsavedarraycomparisons againstbothrefs + fiveCUDA/CUDAjlagreement checks, exactactualinputprefix. Nativehosthistogramcorrectionexplicit; fullphysicsaccuracyoutsidenamedcheckedcontractnotclaimed.'};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['full_artifact_hash_checks','declared_dependency_bindings_checked']},indent=2))
