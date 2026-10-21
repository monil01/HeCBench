#!/usr/bin/env python3
"""Independent read-only postannotation validation; no study imports or subprocesses."""
import argparse,collections,copy,datetime,fcntl,gzip,hashlib,json,re
from pathlib import Path
FIELDS={'model_kernel_regex','semantic_family_in_primary_ROI','ROI_membership_status','ROI_membership_proof','kernel_family_alias_of','alias_of_stage_id','regex_status','resolved_callable_candidates'}
SCOPE_FIELDS={'ROI_stage','semantic_family_in_primary_ROI','sample_phase','ROI_review_sha256','sample_in_primary_timed_invocation'}
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def need(x,s):
 if not x:raise ValueError(s)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--approval-plan',type=Path,required=True);ap.add_argument('--apply-report',type=Path,required=True);ap.add_argument('--strict-baseline',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);a=ap.parse_args()
 root=a.source_root.resolve();study=root/'ontology/studies/v100_roofline_537';out=a.output_root.resolve();need(out.is_relative_to('/tmp')and not out.exists(),'Fresh private output required');out.mkdir(parents=True)
 bindings={};inputs={};result={'status':'not_validated','shared_writes_GPU_build_tests':0,'snapshot_method':'Existing metadata LOCK_SH for whole input snapshot and before/after byte stability; no sample phase assumed.'}
 def read(p):
  p=Path(p);need(p.is_file()and not p.is_symlink(),'Missing/symlink '+str(p));raw=p.read_bytes();need(str(p)not in inputs or inputs[str(p)]==raw,'Input changed inside snapshot');inputs[str(p)]=raw;bindings[str(p)]=sha(raw);return raw
 def obj(p):
  raw=read(p);return json.loads(gzip.decompress(raw)if str(p).endswith('.gz')else raw)
 try:
  with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
   fcntl.flock(lock,fcntl.LOCK_SH)
   approval_raw=read(a.approval_plan);plan=json.loads(approval_raw);applied=obj(a.apply_report)
   need(applied['status']=='applied_journaled_two_renames_pending_annotation_audit','Root apply not successful');need(applied['merge_plan_sha256']==sha(approval_raw),'Apply/approved-plan mismatch');need(applied['transaction']==plan['transaction'],'Transaction mismatch')
   for marker in ('.roi-v4-active-commit.json','.roi-v5-active-commit.json'):
    need(not(study/marker).exists()and not(study/marker).is_symlink(),'Unfinished active transaction '+marker)
   progress=obj(study/'PROGRESS.json');need(datetime.datetime.fromisoformat(progress['updated_at'])>datetime.datetime.fromisoformat(applied['started_at']),'PENDING: PROGRESS predates adopted source annotations')
   need(progress['whole_ROI_classifications']==0 and progress['measured_causal_bottleneck_classifications']==0,'Global whole-ROI/causal promotion')
   target=study/'roi_reviews';backup=Path(applied['rollback_tree']);need(backup.parent==study and backup.name=='.'+plan['transaction']+'-rollback','Unexpected rollback path')
   def tree(p):
    need(p.is_dir()and not p.is_symlink(),'Missing regular directory');need(all(x.is_file()and not x.is_symlink()for x in p.iterdir()),'Unexpected review-tree child');return{x.name:read(x)for x in p.iterdir()}
   old=tree(backup);new=tree(target);need({n:sha(b)for n,b in old.items()}==plan['old_tree'],'Rollback tree differs from approved old bytes');need({n:sha(b)for n,b in new.items()}==plan['new_tree'],'Canonical tree differs from approved new bytes')
   receipt=json.loads(new['enrichment_transaction.json']);need(receipt['transaction']==plan['transaction']and receipt['counts']==plan['counts']and receipt['input_packets']==plan['input_packets'],'Canonical receipt/approval mismatch')
   need(new['enrichment_transaction.json']==read(a.apply_report.parent/'validated_receipt.json'),'Canonical receipt differs from exact root apply receipt')
   documents={};expected={};operations={};counts=collections.Counter();names=set();selected={}
   for pack in plan['input_packets']:
    raw=read(pack['path']);need(sha(raw)==pack['sha256'],'Packet drift');packet=json.loads(raw)
    for b in packet['selected_capture_bindings']:
     key=(b['benchmark'],b['model']);need(key not in selected,'Duplicate model');selected[key]=b
    for patch in packet['patches']:
     key=(patch['record_key']['benchmark'],patch['record_key']['model']);need(key not in operations,'Overlapping packet records');operations[key]=patch;names.add(key[0]);filename=Path(patch['parent_review']['path']).name
     if filename not in documents:documents[filename]=json.loads(gzip.decompress(old[filename]));expected[filename]=copy.deepcopy(documents[filename])
     record=documents[filename]['records'][patch['record_index']];changed=expected[filename]['records'][patch['record_index']];need(canonical(record)==patch['expected_original_record_canonical_sha256'],'Old record SHA mismatch')
     for op in patch['stage_patches']:
      stage=record['stages'][op['stage_index']];need(stage['stage_id']==op['stage_id']and canonical(stage)==op['expected_original_stage_canonical_sha256'],'Old stage SHA mismatch');u=op['updates'];need(set(u)<=FIELDS,'Fields outside whitelist');changed['stages'][op['stage_index']].update(copy.deepcopy(u));counts['stage_updates']+=1
      if 'resolved_callable_candidates'in u:
       science=obj(plan['root_scientific_review']['path']);need(sha(read(plan['root_scientific_review']['path']))==plan['root_scientific_review']['sha256'],'Root scientific review SHA drift');need(science.get('approved_callable_updates',{}).get(op['stage_id'])==u['resolved_callable_candidates'],'Callable update outside exact root stage/value whitelist')
      if op.get('alias_only'):counts['pure_alias_nulls']+=1;need(u.get('model_kernel_regex','x')is None,'Alias selector not null')
      elif u['semantic_family_in_primary_ROI']:counts['inclusion_declarations']+=1;counts['inclusion_alias_nulls']+=int(u.get('model_kernel_regex','x')is None)
      else:counts['exclusions']+=1
   for filename,d in expected.items():need(d==json.loads(gzip.decompress(new[filename])),'Non-whitelisted content changed in '+filename)
   for filename,raw in old.items():
    if filename not in expected and filename not in ('INDEX.json','enrichment_transaction.json'):need(new.get(filename)==raw,'Unrelated review bytes changed '+filename)
   semantic_counts={'model_records':len(operations),'names':len(names),
                    'inclusion_declarations':counts['inclusion_declarations'],'exclusions':counts['exclusions'],
                    'pure_alias_nulls':counts['pure_alias_nulls'],'inclusion_alias_nulls':counts['inclusion_alias_nulls'],
                    'non_null_inclusion_stages':counts['inclusion_declarations']-counts['inclusion_alias_nulls']}
   need(semantic_counts==plan['counts'],'Reconstructed population/aliases differ from approved plan counts')
   # Record-wide and untouched-stage equality follows from whole reconstructed docs above.
   final_records={(r['benchmark'],r['model']):r for d in expected.values()for r in d['records']}
   checks=[];matched_counts=collections.Counter()
   for key,patch in operations.items():
    b,m=key;capture=selected[key];ledger=obj(study/'benchmarks'/(b+'.json'));state=ledger['models'][m];folder=Path(capture['plan']['path']).parent;need(str(folder.relative_to(root))in state['profile_ids'],'Selected profile drift '+str(key));need(state['profiling_status']=='profile_process_completed','Selected status no longer completed')
    need(state.get('roofline_position')is None and state.get('roofline_region')is None and state.get('observed_bottleneck')is None,'Model whole-ROI/causal claim promoted')
    for field in ('plan','derivation','derived','provenance','manifest'):
     if field in capture:need(sha(read(capture[field]['path']))==capture[field]['sha256'],'Selected frozen input changed '+field)
    planpath=folder/'plan.json.gz';profileplan=obj(planpath);need(profileplan['configuration_id']==patch['record_key']['configuration_id'],'Config identity drift')
    for original,keysha in [('result.json','reference_result_sha256'),('config.json','raw_config_sha256')]:need(sha(gzip.decompress(read(study/'selected_contracts'/b/(original+'.gz'))))==profileplan[keysha],'Config/result join drift')
    annotated_path=root/state['kernel_observations']['path'];need(annotated_path.parent==folder,'Annotation from different capture');packed=read(annotated_path);unpacked=gzip.decompress(packed);need(sha(packed)==state['kernel_observations']['sha256']and sha(unpacked)==state['kernel_observations']['uncompressed_sha256'],'Annotation ledger digest mismatch');annotated=json.loads(unpacked)
    rawderived=gzip.decompress(read(folder/'derived.json.gz'));derived=json.loads(rawderived);need(annotated['unannotated_derived_sha256']==sha(rawderived),'Annotated/raw derivation join mismatch')
    reviewfile=Path(patch['parent_review']['path']).name;reviewsha=sha(gzip.decompress(new[reviewfile]));need(annotated['ROI_review_sha256']==reviewsha,'Stale annotation review SHA')
    review=final_records[key];need(not review.get('kernel_mappings'),'Existing exact phase mappings require additional dedicated validation')
    need(len(derived['records'])==len(annotated['records'])==annotated['kernel_count'],'Derived row count changed');modelcounts=collections.Counter()
    for original,row in zip(derived['records'],annotated['records']):
     before=copy.deepcopy(original);after=copy.deepcopy(row);scope0=before.pop('scope_context');scope1=after.pop('scope_context');after.pop('ROI_annotation',None);need(before==after,'Measured counters/math/projections changed '+str(key))
     need({k:v for k,v in scope0.items()if k not in SCOPE_FIELDS}=={k:v for k,v in scope1.items()if k not in SCOPE_FIELDS},'Unexpected scope field changed '+str(key));need(scope1.get('sample_in_primary_timed_invocation')==scope0.get('sample_in_primary_timed_invocation')and scope1.get('sample_in_primary_timed_invocation')is not True,'Actual sample timed flag promoted');need(scope1.get('retained_trial')is None,'Retained trial invented');need(scope1['sample_phase']=='first_source_family_invocation_partial_not_complete_ROI','Actual sample phase promoted')
     matches=[s for s in review['stages']if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],row['identity']['Kernel Name'])];need(len(matches)<=1,'Ambiguous family annotation');proved=bool(matches and matches[0].get('ROI_membership_proof'));expected_semantic=matches[0]['semantic_family_in_primary_ROI']if proved else None;role=row['ROI_annotation'];need(role['stage_candidates']==[s['stage_id']for s in matches]and role['semantic_family_in_primary_ROI']is expected_semantic and scope1['semantic_family_in_primary_ROI']is expected_semantic,'Wrong family membership annotation')
     need(role['mapping_status']==('source_closure_membership_proved_sample_phase_unproved'if proved else'source_callsite_candidate_ROI_unresolved'),'Wrong mapping-strength label');need(scope1['ROI_review_sha256']==reviewsha,'Scope review hash mismatch');need(row.get('causal_bottleneck','unknown')=='unknown','Causal classification promoted')
     modelcounts['sampled_rows']+=1;modelcounts['source_included_rows'if expected_semantic is True else'source_excluded_rows'if expected_semantic is False else'source_unresolved_rows']+=1
    need(modelcounts['source_included_rows']>0,'No inclusion annotation matched this selected capture');matched_counts.update(modelcounts);checks.append({'benchmark':b,'model':m,'selected_profile':str(folder.relative_to(root)),'annotated_sha256':sha(packed),'unannotated_derived_sha256':sha(rawderived),'counts':dict(modelcounts),'sample_phase':'unknown actual invocation; source-family membership only','counters_math_and_projection_values_unchanged':True})
   baseline=obj(a.strict_baseline);strict_raw=read(baseline['path']);need(sha(strict_raw)==baseline['sha256'],'Strict family audit bytes changed');strict=json.loads(strict_raw);eligible={(r['benchmark'],r['model'])for r in strict['records']if r.get('accepted_complete_scientific_kernel_family')};need(eligible=={(b,m)for b in ('triad','babelstream','jacobi')for m in ('cuda','cuda_jl','ka','jacc')}and strict['eligible_family_count']==12,'Strict twelve-family scope changed');need(strict['elapsed_host_ROI_positions']==0 and strict['whole_benchmark_causal_claims']==0,'Strict audit causal/elapsed scope promoted')
   for path,h in plan['source_bindings'].items():need(sha(read(path))==h,'Approved source changed '+path)
   read(Path(__file__).resolve())
   for path,raw in inputs.items():need(Path(path).read_bytes()==raw,'Input changed during SH snapshot '+path)
   result.update(status='postannotation_validation_passed',progress_updated_at=progress['updated_at'],transaction=plan['transaction'],counts=dict(counts,model_records=len(operations),names=len(names),non_null_inclusion_stages=counts['inclusion_declarations']-counts['inclusion_alias_nulls']),selected_derived_counts=dict(matched_counts),models=checks,strict_family_audit={'sha256':sha(strict_raw),'count':12,'unchanged':True},whole_ROI_classifications=0,causal_classifications=0,actual_sample_timed_flag_promotions=0,source_family_claim_only=True,approval_plan_sha256=sha(approval_raw),bindings=bindings)
  rc=0
 except Exception as error:result.update(status='pending_or_validation_failed',error=repr(error),bindings=bindings);rc=2 if str(error).startswith('PENDING:')else 1
 (out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':result['status'],'error':result.get('error'),'counts':result.get('counts'),'selected_derived_counts':result.get('selected_derived_counts'),'validation_sha256':sha((out/'validation.json').read_bytes()),'output':str(out)}));return rc
if __name__=='__main__':raise SystemExit(main())
