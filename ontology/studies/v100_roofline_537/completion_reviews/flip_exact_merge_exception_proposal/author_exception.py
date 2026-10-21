from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent;P=Path('/tmp/roofline-completion-20261021/roi_membership_root_merge_v5_preparation/root_merge_reviews_v5_callable_latin1.py');old=P.read_text();base='a05280245cfd0297e459a34e107f460f50bc4df1ce1f773d570a210bd9f0c76a'
if hashlib.sha256(P.read_bytes()).hexdigest()!=base:raise ValueError('Exact root a052 base required')
v=old
def replace(a,b):
 global v
 if v.count(a)!=1:raise ValueError('Source anchor mismatch')
 v=v.replace(a,b,1)
replace(' receipts=[];source_hashes={};source_cache={};overlay_checks=[];identities=set()',' receipts=[];source_hashes={};source_cache={};overlay_checks=[];identities=set();configuration_id_join_exceptions=[]')
anchor="  for k,ck in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]:need(plan[k]==contract[ck],'Selected contract join '+k)"
new='''  for k,ck in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]:
   if k=='configuration_id'and plan[k]!=contract[ck]:
    # Exact historic flip schema exception ONLY. Neither the canonical null
    # nor the immutable result/config is filled in or otherwise modified.
    expected={'approved':True,'configuration_id':'flip_deterministic_n4096_r100_v1','canonical_configuration_id':None,'result_configuration_id_field_absent':True,'raw_config_sha256':'371c61a3cac140641d17e038c9118551a86626adc61201caf9d00b2c7f283a9c','reference_result_sha256':'803e84790e80a9bc706967cb8633c542a83eeba0f00d29eff9ddf2d9a3477e7d'}
    need(b=='flip'and m in ('cuda','cuda_jl','ka','jacc')and contract[ck]is None,'No generic configuration-ID fallback permitted')
    need(scientific.get('approved_contract_join_exceptions',{}).get('flip')==expected,'Exact hash-bound root-note approval of flip exception missing')
    configraw=gzip.decompress(file_bytes(study/'selected_contracts/flip/config.json.gz'));resultraw=gzip.decompress(file_bytes(study/'selected_contracts/flip/result.json.gz'))
    configdoc=json.loads(configraw);resultdoc=json.loads(resultraw)
    need(sha(configraw)==expected['raw_config_sha256']==plan['raw_config_sha256']==contract['raw_config_sha256'],'Exact flip frozen config identity mismatch')
    need(sha(resultraw)==expected['reference_result_sha256']==plan['reference_result_sha256']==contract['result_sha256'],'Exact flip frozen result identity mismatch')
    need(configdoc.get('benchmark')==resultdoc.get('benchmark')=='flip'and'configuration_id'not in resultdoc,'Flip original result must lack CID rather than conflict')
    need(configdoc.get('configuration_id')==plan[k]==record['configuration_id']==patch['record_key']['configuration_id']==expected['configuration_id'],'Exact flip frozen-config/plan/review/proposal ID agreement required')
    configuration_id_join_exceptions.append({'benchmark':b,'model':m,'root_scientific_review':scientific_binding,'root_approval':expected,'plan_sha256':sha(planraw),'profile_provenance_sha256':sha(provraw),'canonical_configuration_id_unchanged':None,'historical_result_configuration_id_field_remains_absent':True,'interpretation':'Hash-bound identity join only; no metadata fill, acceptance, execution or sample-phase change.'})
   else:need(plan[k]==contract[ck],'Selected contract join '+k)'''
replace(anchor,new)
replace("'retained_AMG_overlay_checks':[x for x in overlay_checks", "'configuration_id_join_exceptions':configuration_id_join_exceptions,'retained_AMG_overlay_checks':[x for x in overlay_checks")
(D/'root_a052_original.py').write_bytes(P.read_bytes());(D/'merge_flip_exact.py').write_text(v);(D/'minimal.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),v.splitlines(True),fromfile=str(P),tofile='private/merge_flip_exact.py')))
expected={'approved':True,'configuration_id':'flip_deterministic_n4096_r100_v1','canonical_configuration_id':None,'result_configuration_id_field_absent':True,'raw_config_sha256':'371c61a3cac140641d17e038c9118551a86626adc61201caf9d00b2c7f283a9c','reference_result_sha256':'803e84790e80a9bc706967cb8633c542a83eeba0f00d29eff9ddf2d9a3477e7d'}
(D/'root_note_required_field.example.json').write_text(json.dumps({'NOT_AN_APPROVAL':'Root must review and author its own complete note; this file cannot replace required packets/counts/science review.','approved_contract_join_exceptions':{'flip':expected}},indent=2)+'\n')
(D/'authoring.json').write_text(json.dumps({'base_path':str(P),'base_sha256':base,'candidate_sha256':hashlib.sha256(v.encode()).hexdigest(),'candidate_imported_executed_tested_or_applied':False,'canonical_and_historical_bytes_changed':False,'scope':'Only exact flip mismatch null versus known literal ID with fixed config/result hashes and original result CID absent; exact rootnote dictionary required; all ordinary joins remain.'},indent=2)+'\n')
print(hashlib.sha256(v.encode()).hexdigest())
