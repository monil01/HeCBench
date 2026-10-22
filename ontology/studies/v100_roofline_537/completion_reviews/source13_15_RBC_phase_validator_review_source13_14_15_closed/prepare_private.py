from pathlib import Path
import json,gzip,hashlib,fcntl,difflib
O=Path(__file__).parent;W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';base=Path('/tmp/roofline-completion-20261021/roi_batch4_postannotation_validator/root_validate_postannotation_callable.py');H=lambda b:hashlib.sha256(b).hexdigest();pins={};findings=[];fields=['sample_in_primary_timed_invocation','retained_trial','sample_phase'];triplet={'sample_in_primary_timed_invocation':True,'retained_trial':2,'sample_phase':'source_checked_trial2_step240_partial_specializations'}
with open('/tmp/hecbench-roofline-metadata.lock','r')as l:
 fcntl.flock(l,fcntl.LOCK_SH);ledger=json.loads((S/'benchmarks/rayleighBenardConvection.json').read_bytes())
 for m in ['cuda','cuda_jl','ka','jacc']:
  st=ledger['models'][m];folder=(W/st['profile_evidence_manifest']).parent;manifest=json.loads((folder/'manifest.json').read_bytes());files={str(folder.relative_to(W))+'/manifest.json':H((folder/'manifest.json').read_bytes())}
  for f in manifest['files']:
   raw=(W/f['path']).read_bytes();assert H(raw)==f['compressed_sha256'];assert H(gzip.decompress(raw))==f['uncompressed_sha256'];files[f['path']]=H(raw)
  derived=json.loads(gzip.decompress((folder/'derived.json.gz').read_bytes()));annotated=json.loads(gzip.decompress((W/st['kernel_observations']['path']).read_bytes()));prov=json.loads(gzip.decompress((folder/'profile_provenance.json.gz').read_bytes()));pl=json.loads(gzip.decompress((folder/'plan.json.gz').read_bytes()));reviewraw=gzip.decompress((folder/'profile_window_review.json.gz').read_bytes());review=json.loads(reviewraw);wr=prov['profile_window'];assert H(reviewraw)==wr['review']['sha256'];assert review['models'][wr['review']['entry']]==wr['record'];assert prov['profile_window_runtime_checks']['all_pass']is True and prov['profile_window_runtime_checks']['checked_trial_count']==5;assert len(derived['records'])==len(annotated['records'])==27
  for a,b in zip(derived['records'],annotated['records']):assert {k:a['scope_context'][k]for k in fields}==triplet=={k:b['scope_context'][k]for k in fields}
  op=prov['operation_calibration'];assert H((W/op['path']).read_bytes())==op['sha256'];files[op['path']]=op['sha256'];pins[m]={'folder':str(folder.relative_to(W)),'files':files};findings.append({'model':m,'folder':str(folder),'rows':27,'phase_triplet_raw_equals_annotated':True,'raw_triplet':triplet,'profile_window_review_SHA':H(reviewraw),'window':wr['record']['selected_window'],'runtime_checks':prov['profile_window_runtime_checks'],'configuration_id':pl['configuration_id'],'result_sha256':pl['reference_result_sha256'],'config_sha256':pl['raw_config_sha256'],'source_bound_marker_authority':wr['record']['source_changes'][-2:],'GPU_bodies_original_equals_copy':wr['record']['original_GPU_function_bytes_sha256']==wr['record']['copied_GPU_function_bytes_sha256']})
 fcntl.flock(l,fcntl.LOCK_UN)
old=base.read_text();(O/'base_validator.py').write_text(old)
helper='''# Exact existing four RBC captures only; no generic phase-authority relaxation.
RBC_RETAINED_CAPTURE_PINS = '''+repr(pins)+'''
def preserved_rbc_phase(key, folder, root, profileplan, derived, read, obj):
 if key[0] != 'rayleighBenardConvection' or key[1] not in RBC_RETAINED_CAPTURE_PINS:return False
 pin=RBC_RETAINED_CAPTURE_PINS[key[1]]
 # A different capture is not admitted by this exception.
 if folder != root/pin['folder']:return False
 for relative,h in pin['files'].items():need(sha(read(root/relative))==h,'Exact RBC retained-phase authority changed: '+relative)
 need(profileplan['configuration_id']=='original500x1000_1000steps_fullRK3_cuBLAS_excessNu_qualified_v1' and profileplan['reference_result_sha256']=='dba26f95fa4e5d3103fe0131f535c043a551fe3a0237d5f5bf7a5c1160cfae5b' and profileplan['raw_config_sha256']=='ff4e7765f0bb87f28bed25abd68547a503eb61e645732dd00aa29010b1ad66f3','RBC named contract drift')
 prov=obj(folder/'profile_provenance.json.gz');review_raw=gzip.decompress(read(folder/'profile_window_review.json.gz'));review=json.loads(review_raw);window=prov['profile_window'];checks=prov['profile_window_runtime_checks']
 need(sha(review_raw)==window['review']['sha256'] and review['models'][window['review']['entry']]==window['record'],'RBC saved review/phase authority join')
 need(prov['profile']['returncode']==0 and checks['all_pass'] is True and checks['checked_trial_count']==5 and len(checks['begin'])==len(checks['end'])==1,'RBC retained full five checks/markers required')
 need(checks['begin'][0]['trial']==2 and checks['begin'][0]['zero_based_step']==240 and checks['end'][0]['authored_dispatch_counts']==[3,6,2,11,5,0,1] and checks['end'][0]['whole_1000step_capture'] is False,'RBC retained window drift')
 need(len(derived['records'])==27,'Exact RBC saved row count changed')
 return True
'''
oldscope="need(scope1.get('sample_in_primary_timed_invocation')==scope0.get('sample_in_primary_timed_invocation')and scope1.get('sample_in_primary_timed_invocation')is not True,'Actual sample timed flag promoted');need(scope1.get('retained_trial')is None,'Retained trial invented');need(scope1['sample_phase']=='first_source_family_invocation_partial_not_complete_ROI','Actual sample phase promoted')"
newscope="""phase_fields=('sample_in_primary_timed_invocation','retained_trial','sample_phase')
     if preserved_phase:
      need(all(k in scope0 and k in scope1 for k in phase_fields),'RBC existing phase field missing')
      expected_phase={'sample_in_primary_timed_invocation':True,'retained_trial':2,'sample_phase':'source_checked_trial2_step240_partial_specializations'}
      need(canonical({k:scope0[k]for k in phase_fields})==canonical(expected_phase)==canonical({k:scope1[k]for k in phase_fields}),'RBC pre-existing phase fields changed, not preserved')
     else:
      need(scope1.get('sample_in_primary_timed_invocation')==scope0.get('sample_in_primary_timed_invocation')and scope1.get('sample_in_primary_timed_invocation')is not True,'Actual sample timed flag promoted');need(scope1.get('retained_trial')is None,'Retained trial invented');need(scope1['sample_phase']=='first_source_family_invocation_partial_not_complete_ROI','Actual sample phase promoted')"""
assert old.count(oldscope)==1
new=old.replace('def main():',helper+'\ndef main():',1).replace("for original,row in zip(derived['records'],annotated['records']):","preserved_phase=preserved_rbc_phase(key,folder,root,profileplan,derived,read,obj)\n    for original,row in zip(derived['records'],annotated['records']):",1).replace(oldscope,newscope,1)
(O/'proposed_validator.py').write_text(new);(O/'minimal.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(base),tofile='PRIVATE/proposed_validator.py')));out={'source_base':{'path':str(base),'sha256':H(old.encode())},'proposed_source_sha256':H(new.encode()),'scope':'Private source-only validator proposal; not executed/imported/tested; no shared changes','root_adoption_required':True,'exact_RBC_models':findings,'preserved_existing_rows':108,'original_failed_report':{'path':'/tmp/roofline-completion-20261021/roi_batch13_14_15_postannotation_actual_001/validation.json','sha256':H(Path('/tmp/roofline-completion-20261021/roi_batch13_14_15_postannotation_actual_001/validation.json').read_bytes())},'no_new_phase_proof':True,'complete_ROI_claim':False,'causal_claim':False};(O/'review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'base_sha256':H(old.encode()),'new_sha256':H(new.encode()),'review_sha256':H((O/'review.json').read_bytes()),'rows':108,'proposal_not_executed':True}))
