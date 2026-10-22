from pathlib import Path
import json,gzip,hashlib,re,fcntl,datetime
O=Path(__file__).parent;D=O.with_name('roi_membership_source_batch13_qr');W=Path('/home/imo/HeCBench_julia');P=D/'proposed_stage_patches.json';sha=lambda b:hashlib.sha256(b).hexdigest();H=lambda p:sha(Path(p).read_bytes());canon=lambda x:sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode());d=json.loads(P.read_bytes());checks=[];issues=[]
def ck(name,yes,detail=None):
 checks.append({'check':name,'passed':bool(yes),'detail':detail})
 if not yes:issues.append({'check':name,'detail':detail})
ck('exact_reviewed_packet',H(P)=='d167267d609d87d7db5cc30bfe8cc8354a17013d6fa5283bfdef8b00d7560d45')
ck('30_exact_models',len(d['patches'])==30)
ck('exact_holds',{(r['benchmark'],r['model'])for r in d['withheld_models']}=={('radixsort2','cuda'),('rainflow','cuda')})
ck('phase_and_fullROI_false_top',d['sample_phase_still_unproved']is True and d['elapsed_ROI_aggregate_claim']is False and d['causal_claim']is False)
parents={};sources={};captures={};matched=0;unassigned=0
proof=json.loads(gzip.decompress((D/'proofpacket.json.gz').read_bytes()))
quote_count=0
for path,ev in proof['sources'].items():
 raw=Path(path).read_bytes();enc=ev.get('snippet_encoding','utf-8');ck('quote_source_sha',sha(raw)==ev['sha256'],path);ck('exact_encoding_allowlist',enc=='utf-8',path);ck('full_snippet_byte_inverse',ev['snippet'].encode(enc)==raw[ev['start_byte']:ev['end_byte']],path);ck('full_source_span',ev['start_byte']==0 and ev['end_byte']==len(raw) and ev['start_line']==1 and ev['end_line']==ev['snippet'].count('\n')+1,path);quote_count+=1
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for path,h in d['parent_reviews'].items():
  raw=Path(path).read_bytes();ck('parent_compressed_hash',sha(raw)==h['compressed_sha256'],path);u=gzip.decompress(raw);ck('parent_uncompressed_hash',sha(u)==h['uncompressed_sha256'],path);parents[path]=json.loads(u)
 for row in d['sources']:ck('present_source_hash',H(row['path'])==row['sha256'],row['path']);sources[row['path']]=row['sha256']
 for row in d['selected_capture_bindings']:
  for k in ['manifest','plan','derivation','provenance']:ck('selected_capture_hash',H(row[k]['path'])==row[k]['sha256'],{'benchmark':row['benchmark'],'model':row['model'],'kind':k})
  captures[(row['benchmark'],row['model'])]=row
  ledger=json.loads(Path(row['ledger']['path']).read_bytes());state=ledger['models'][row['model']];manifest_path=(W/state['profile_evidence_manifest']).resolve();ck('current_selected_capture_identity',manifest_path==Path(row['manifest']['path']).resolve(),[row['benchmark'],row['model']]);plan=json.loads(gzip.decompress(Path(row['plan']['path']).read_bytes()));contract=ledger['accepted_named_contract'];ck('current_contract_CID_config_result',all(plan[a]==contract[b]for a,b in [('configuration_id','configuration_id'),('raw_config_sha256','raw_config_sha256'),('reference_result_sha256','result_sha256')]),[row['benchmark'],row['model']])
 for r in d['patches']:
  key=r['record_key'];old=parents[r['parent_review']['path']]['records'][r['record_index']];tag=key['benchmark']+'/'+key['model'];ck('exact_parent_key_index',all(old[k]==v for k,v in key.items()),tag);ck('canonical_record_hash',canon(old)==r['expected_original_record_canonical_sha256'],tag)
  c=captures[(key['benchmark'],key['model'])];pl=json.loads(gzip.decompress(Path(c['plan']['path']).read_bytes()));ob=json.loads(gzip.decompress(Path(c['derivation']['path']).read_bytes()))['records'];ck('selected_CID_config_result_joins',pl['configuration_id']==old['configuration_id']and pl['raw_config_sha256']==old['raw_config_sha256']and pl['reference_result_sha256']==old['reference_result']['sha256'],tag)
  for k,v in r['unchanged_record_qualifications'].items():ck('unchanged_record_qualifier',old[k]==v,{'slot':tag,'key':k})
  new=[dict(s)for s in old['stages']]
  for sp in r['stage_patches']:
   st=old['stages'][sp['stage_index']];ck('canonical_stage_hash',st['stage_id']==sp['stage_id']and canon(st)==sp['expected_original_stage_canonical_sha256'],tag);u=sp['updates'];ck('source_only_update_fields',{'semantic_family_in_primary_ROI','ROI_membership_status','ROI_membership_proof'}<=set(u)<= {'semantic_family_in_primary_ROI','ROI_membership_status','ROI_membership_proof','model_kernel_regex','regex_status','alias_of_stage_id'},tag);p=u['ROI_membership_proof'];ck('source_inclusion_only',(u['semantic_family_in_primary_ROI']is True and u['ROI_membership_status']=='source_timed_closure_inclusion_proved')or(u['semantic_family_in_primary_ROI']is False and u['ROI_membership_status']in ['source_setup_exclusion_proved','source_untimed_audit_exclusion_proved']),tag)
   ck('proof_no_phase_or_causal_promotion',p['sample_in_primary_timed_invocation']is not True and p['retained_trial']is None and p['whole_elapsed_ROI_claim']is False and p['complete_ROI']is False and p['causal_bottleneck']=='unknown',tag)
   ck('proof_selected_capture_join',p['selected_result']==old['reference_result']and p['selected_config']==old['raw_config']and p['actual_profile_source_bindings']['sha256']==c['plan']['sha256'],tag)
   for ev in p['evidence']:
    ck('proof_evidence_source_hash',sources.get(ev['path'])==ev['sha256'],{'slot':tag,'path':ev['path']})
    lines=Path(ev['path']).read_bytes().decode('utf-8').splitlines()
    for line in ev.get('line_evidence',[]):ck('exact_line_quote',lines[line['line']-1]==line['text'],{'slot':tag,'path':ev['path'],'line':line['line']})
   new[sp['stage_index']].update(u)
  actualmatches=[]
  for x in ob:
   hits=[s['stage_id']for s in new if s.get('semantic_family_in_primary_ROI')is True and s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],x['identity']['Kernel Name'])];ck('unique_actual_saved_family_match',len(hits)<=1,tag);actualmatches.append((x['identity'],hits));matched+=bool(hits);unassigned+=not bool(hits)
  ck('model_positive_saved_match',any(v for _,v in actualmatches),tag)
  supplied=[(x['identity'],x['matched_stage_ids'])for x in r['selected_observation_matching']];ck('recomputed_saved_match_equality',actualmatches==supplied,tag)
 fcntl.flock(lock,fcntl.LOCK_UN)
report={'packet':{'path':str(P),'sha256':H(P)},'adoption_admissible_subject_to_root_atomic_parent_revalidation':not issues,'blockers':issues,'checks_total':len(checks),'checks_passed':sum(x['passed']for x in checks),'checks':checks,'saved_matching_recomputed':{'matched':matched,'unassigned':unassigned,'models':len(d['patches'])},'runtime_sample_phase':'unproved','whole_ROI_claim':False,'causal_claim':False,'full_source_byte_inverse_checks':quote_count,'GPU_runs':0,'build_runs':0,'tests':0,'shared_mutations':False}
q=O/'independent_review.json';q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'admissible':not issues,'checks':len(checks),'issues':issues,'matches':matched,'unassigned':unassigned,'sha256':H(q)}))
