from pathlib import Path
import json,gzip,hashlib,fcntl,re,copy,datetime
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';P=Path('/tmp/roofline-completion-20261021/roi_membership_batch12_stencil/proposed_stage_patches.json');O=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest();canon=lambda x:h(json.dumps(x,sort_keys=True,separators=(',',':')).encode());inputs={};sourcebytes={};quoted_count=0
ALLOWED={'model_kernel_regex','semantic_family_in_primary_ROI','ROI_membership_status','ROI_membership_proof','kernel_family_alias_of','alias_of_stage_id','regex_status','resolved_callable_candidates'}
def read(p):
 p=Path(p);b=p.read_bytes();inputs[str(p)]={'sha256':h(b),'bytes':len(b)};return b
def load(p):
 b=read(p);return json.loads(gzip.decompress(b)if str(p).endswith('.gz')else b)
def bound(e):
 b=read(e['path']);assert h(b)==(e.get('sha256')or e.get('compressed_sha256'));return b
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);raw=read(P);assert h(raw)=='89b1b9201b4fac3983ad4743c5a5b7c01953a81029fd6a1e0bc71aa421337d74';a=json.loads(raw);indexraw=read(S/'roi_reviews/INDEX.json');index=json.loads(indexraw);assert bound(a['parent_INDEX'])==indexraw
 assert a['sample_phase_still_unproved']is True and a['elapsed_ROI_aggregate_claim']is False and a['causal_claim']is False
 leg=a['legacy_authority_exclusion_check'];lb=read(leg['path']);assert h(lb)==leg['compressed_sha256']and h(gzip.decompress(lb))==leg['uncompressed_sha256'];bound(leg['receipt']);covered={r['benchmark']for r in json.loads(gzip.decompress(lb))['records']};parents={}
 for x in index['cohorts']:
  p=W/x['path'];b=read(p);un=gzip.decompress(b);assert h(b)==x['compressed_sha256']and h(un)==x['raw_sha256'];doc=json.loads(un);parents[str(p)]=(b,un,doc)
  for r in doc['records']:
   if any(str(s.get('ROI_membership_status','')).startswith('source_')or s.get('ROI_membership_proof')for s in r['stages']):covered.add(r['benchmark'])
 names={r['record_key']['benchmark']for r in a['patches']};assert not names&covered
 for e in a['sources']:sourcebytes[str(Path(e['path']).resolve())]=bound(e)
 caps={(x['benchmark'],x['model']):x for x in a['selected_capture_bindings']};assert len(caps)==len(a['patches'])==62;allrows=[];countinc=countexc=matched=total=0;corrections=[]
 for patch in a['patches']:
  key=patch['record_key'];b,m=key['benchmark'],key['model'];p=patch['parent_review'];packed,un,doc=parents[p['path']];assert h(packed)==p['compressed_sha256']and h(un)==p['uncompressed_sha256'];record=doc['records'][patch['record_index']];assert (record['benchmark'],record['model'],record['configuration_id'])==(b,m,key['configuration_id']);assert canon(record)==patch['expected_original_record_canonical_sha256'];assert sum((r['benchmark'],r['model'])==(b,m)for r in doc['records'])==1
  for k,v in patch['unchanged_record_qualifications'].items():assert record[k]==v
  cap=caps[b,m];ledger=json.loads(bound(cap['ledger']));plan=json.loads(gzip.decompress(bound(cap['plan'])));derived=json.loads(gzip.decompress(bound(cap['derivation'])));manifest=json.loads(bound(cap['manifest']));folder=Path(cap['plan']['path']).parent
  members={}
  for name in ['plan.json','derived.json','profile_provenance.json']:
   es=[e for e in manifest['files']if e['original_name']==name];assert len(es)==1;e=es[0];pb=read(W/e['path']);ub=gzip.decompress(pb);assert h(pb)==e['compressed_sha256']and h(ub)==e['uncompressed_sha256']and len(pb)==e['compressed_bytes']and len(ub)==e['uncompressed_bytes'];members[name]=(json.loads(ub),ub)
  assert plan==members['plan.json'][0]and derived==members['derived.json'][0];prov=members['profile_provenance.json'][0];assert prov['status']=='profile_process_completed'and prov['profile']['returncode']==0 and prov['profile'].get('stop_reason')is None;assert prov['plan_sha256']==h(members['plan.json'][1])
  state=ledger['models'][m];contract=ledger['accepted_named_contract'];assert state['profiling_status']=='profile_process_completed';assert str(folder.relative_to(W))in state['profile_ids'];assert state['profile_provenance_sha256']==h(members['profile_provenance.json'][1]);assert (W/state['kernel_observations']['path']).parent==folder
  for pk,ck in [('configuration_id','configuration_id'),('raw_config_sha256','raw_config_sha256'),('reference_result_sha256','result_sha256')]:assert plan[pk]==contract[ck]
  assert plan['configuration_id']==record['configuration_id']and plan['raw_config_sha256']==record['raw_config_sha256']and plan['reference_result_sha256']==record['reference_result']['sha256'];bound(record['reference_result']);bound(record['raw_config'])
  capture={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()};capture.update({str(Path(k).resolve()):v for k,v in prov.get('additional_current_profile_input_bindings',{}).items()})
  def quote(z):
   global quoted_count
   path=str(Path(z['path']).resolve());rb=sourcebytes.get(path)
   if rb is None:rb=read(path)
   sh=z.get('source_sha256')or z.get('sha256');assert h(rb)==sh==capture[path];text=rb.decode();assert text.encode()==rb;lines=text.splitlines();quoted_count+=1
   if 'line'in z and 'text'in z:assert lines[z['line']-1]==z['text']
   if 'context'in z:assert '\n'.join(lines[z['context_start_line']-1:]).startswith(z['context'])
   if 'exact_lines'in z:assert '\n'.join(lines[z['start_line']-1:z['end_line']])==z['exact_lines']
   for label in ['timer_lines','arithmetic_branch_atomic_lines','line_evidence']:
    for q in z.get(label,[]):assert lines[q['line']-1]==q['text']
   if 'snippet'in z:
    start,end=z['start_byte'],z['end_byte'];assert type(start)is int and type(end)is int and 0<=start<=end<=len(rb);assert rb[start:end].decode()==z['snippet']and z['snippet'].encode()==rb[start:end]
    if 'start_line'in z:assert rb[:start].count(b'\n')+1==z['start_line']
    if 'end_line'in z:assert z['end_line']in {rb[:end].count(b'\n')+1,rb[:max(start,end-1)].count(b'\n')+1}
  revised=copy.deepcopy(record);seen=set()
  for st in patch['stage_patches']:
   old=record['stages'][st['stage_index']];u=st['updates'];assert old['stage_id']==st['stage_id']and canon(old)==st['expected_original_stage_canonical_sha256'];assert st['stage_id']not in seen;seen.add(st['stage_id']);assert set(u)<=ALLOWED;inc=u['semantic_family_in_primary_ROI'];assert type(inc)is bool;assert u['ROI_membership_status']in (['source_timed_closure_inclusion_proved']if inc else ['source_setup_exclusion_proved','source_untimed_audit_exclusion_proved']);countinc+=inc;countexc+=not inc
   pr=u['ROI_membership_proof'];assert pr.get('sample_in_primary_timed_invocation')in [None,False]and pr.get('retained_trial')is None and not pr.get('whole_elapsed_ROI_claim',False)and not pr.get('complete_ROI',False)and pr.get('causal_bottleneck','unknown')in ['unknown',None];assert pr['profile_target_command']==plan['profile_target_command']
   for z in [pr['exact_stage_callsite']]+pr.get('evidence',[])+pr.get('timed_closure_boundaries',[]):quote(z)
   for label in ['selected_result','selected_config','selected_counter_derivation','selected_profile_manifest','actual_profile_source_bindings']:
    if label in pr:bound(pr[label])
   if 'resolved_callable_candidates'in u:corrections.append({'benchmark':b,'model':m,'stage_id':st['stage_id'],'value':u['resolved_callable_candidates']})
   revised['stages'][st['stage_index']].update(copy.deepcopy(u))
  assert {k:v for k,v in record.items()if k!='stages'}=={k:v for k,v in revised.items()if k!='stages'}
  matches=[]
  for ob in derived['records']:
   hits=[s for s in revised['stages']if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],ob['identity']['Kernel Name'])];assert len(hits)<=1;assert not hits or hits[0]['semantic_family_in_primary_ROI']is True;matches.append({'identity':ob['identity'],'matched_stage_ids':[s['stage_id']for s in hits]})
  assert any(x['matched_stage_ids']for x in matches)
  expected=patch['selected_observation_matching'];assert len(matches)==len(expected)
  for x,y in zip(matches,expected):assert x['identity']==y['identity']and x['matched_stage_ids']==y['matched_stage_ids']and y['runtime_first_sample_phase']=='unproved'
  nmatch=sum(bool(x['matched_stage_ids'])for x in matches);matched+=nmatch;total+=len(matches);allrows.append({'benchmark':b,'model':m,'stages':len(patch['stage_patches']),'saved_samples':len(matches),'matched':nmatch,'positive_included_family':True,'exact_selected_contract_capture_source_stage_parent_joins':True})
 assert (countinc,countexc,matched,total)==(155,67,368,388);assert corrections==[{'benchmark': 'moe-sum', 'model': 'jacc', 'stage_id': 'moe-sum/jacc/source_callsite_001', 'value': ['moe_vec4_jacc!']}, {'benchmark': 'multinomial', 'model': 'cuda_jl', 'stage_id': 'multinomial/cuda_jl/source_callsite_001', 'value': ['sample_multinomial_once_kernel!']}, {'benchmark': 'muon', 'model': 'cuda_jl', 'stage_id': 'muon/cuda_jl/source_callsite_003', 'value': ['sumsq_cuda!']}, {'benchmark': 'muon', 'model': 'cuda_jl', 'stage_id': 'muon/cuda_jl/source_callsite_004', 'value': ['normalize_cuda!']}, {'benchmark': 'muon', 'model': 'cuda_jl', 'stage_id': 'muon/cuda_jl/source_callsite_005', 'value': ['combine_cuda!']}, {'benchmark': 'muon', 'model': 'cuda_jl', 'stage_id': 'muon/cuda_jl/source_callsite_006', 'value': ['update_cuda!']}, {'benchmark': 'muon', 'model': 'jacc', 'stage_id': 'muon/jacc/source_callsite_001', 'value': ['momentum_transpose_jacc!', 'momentum_jacc!', 'sumsq_jacc!', 'normalize_jacc!', 'combine_jacc!', 'update_jacc!', 'transpose_parameter_jacc!', 'parameter_jacc!']}]
 for path,x in inputs.items():assert h(Path(path).read_bytes())==x['sha256']
 assert read(P)==raw
 report={'status':'independent_source_and_saved_capture_review_consistent_pending_root_review','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proposal':{'path':str(P),'sha256':h(raw)},'counts':{'models':62,'names':16,'stages':222,'inclusions':countinc,'exclusions':countexc,'samples':total,'matched':matched,'unassigned':total-matched},'required_exact_root_callable_approval':corrections,'quoted_evidence_checks':quoted_count,'current_INDEX_sha256':h(indexraw),'all_reviewed_input_bytes_stable':True,'model_records':allrows,'input_bindings':inputs,'no_candidate_execution_import_or_GPU':True,'scope':'Source-family membership only; original scientific/timing/coverage qualifiers preserved. No actualphase/fullROI/causal promotion.'}
 (O/'review.json').write_text(json.dumps(report,indent=2)+'\n');(O/'parent_INDEX_at_review.json').write_bytes(indexraw);print(json.dumps({k:v for k,v in report.items()if k not in ['input_bindings','model_records']},indent=2))
