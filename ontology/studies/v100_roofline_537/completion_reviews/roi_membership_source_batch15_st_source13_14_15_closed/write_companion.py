from pathlib import Path
import json,gzip,hashlib,fcntl,datetime,re
O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';H=lambda b:hashlib.sha256(b).hexdigest();p=json.loads((O/'proposed_stage_patches.json').read_bytes());proof=json.loads(gzip.decompress((O/'proofpacket.json.gz').read_bytes()))
def evidence(rel,needles):
 path=str(W/rel);d=proof['sources'][path];b=Path(path).read_bytes();assert H(b)==d['sha256'];lines=b.decode(d['snippet_encoding']).splitlines(keepends=True);ev=[]
 for needle in needles:
  ids=[i for i,line in enumerate(lines)if needle in line];assert ids,(path,needle)
  for i in ids:
   st=sum(len(l.encode(d['snippet_encoding']))for l in lines[:i]);raw=lines[i].encode(d['snippet_encoding']);assert b[st:st+len(raw)]==raw;ev.append({'path':path,'sha256':H(b),'line':i+1,'start_byte':st,'end_byte':st+len(raw),'snippet_encoding':d['snippet_encoding'],'text':lines[i]})
 return ev
obs=[{'benchmark':'s3d','observation':'The selected f64 chemistry-rate pipeline evaluates27stages over4096uniform synthetic cells with fixed1000temperature,1.0132e6pressure and unitmolecularweights.100wholepipeline launch/contextcompletion repeats are averaged.508sourceCPUformula values percell validate fullheterogeneous arrays outside the timer; this is not independent chemistry, coupledflow or timeintegration validation.','evidence':evidence('src/s3d-julia/study_common.jl',['fp64=ARGS','ht=fill','for rep in 1:R','heterogeneous_full_WDOT']),'scientific_intent':proof['scientific_intent']['s3d']},{'benchmark':'sa','observation':'The Int32recursiveDC3 suffix-array operator retains Thrust CABI allocation/sort/scan and host rankfeedback, alongside authored framework kernels. Timed constructions include assignment/H2D, recursive temporary allocation/free, customkernels, vendor sort/scan and D2H; outercasecontainer creation/deletion and independent permutation/finite-tail lexorder certification are outside the interval. Only final construction outputs pertrial are certified/archived, not every internalrepeat. Sourcefamily inclusion cannot establish sampled runtimephase or completevendorcountercoverage.','evidence':evidence('src/sa-julia/study_common.jl',['sa_sort(k','sa_scan(h','maxrank=sa_scalar','sa_new(0)','for it=1:repeats','elapsed=(time_ns()-t)','all_internal_iteration_outputs_archived']),'scientific_intent':proof['scientific_intent']['sa']},{'benchmark':'sc','observation':'The named Int32compaction timer contains oneGPUlaunch plus4CPUworker start/contextcompletion/join for each15staticpartition cases. Reset, padding checks, posttimer outputmerge and complete count/flag/guard oracle remain outside. This mixed host/device operator cannot be represented as wholeelapsedROI using its one devicefamily or interpreted as a causal framework/allocator comparison.','evidence':evidence('src/sc-julia/study_common.jl',['cut=trunc','function run(c)','ccall(startfn','ccall(waitfn','start=time_ns();run(c)','results==oracle']),'scientific_intent':proof['scientific_intent']['sc']}]
(O/'companion_scientific_observations.json').write_text(json.dumps({'scope':'Exact-source scientific distinctions; no numerical recertification/runtimephase/fullROI/causal claim','observations':obs},indent=2)+'\n')
checks=0
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);assert(S/'roi_reviews/INDEX.json').read_bytes()==(O/'selection_parent_INDEX.json').read_bytes();sel=json.loads((O/'selection_authority_review.json').read_bytes());assert H(Path(sel['legacy']['path']).read_bytes())==sel['legacy']['sha256'];checks+=2
 for d in proof['sources'].values():
  b=Path(d['path']).read_bytes();enc=d['snippet_encoding'];assert H(b)==d['sha256']and b[d['start_byte']:d['end_byte']]==d['snippet'].encode(enc);assert d['end_line']==b.decode(enc).count('\n')+1;checks+=1
 for row in p['patches']:
  for op in row['stage_patches']:
   st=op['updates']['ROI_membership_proof']['exact_stage_callsite'];path=str(Path(st['path']).resolve());b=Path(path).read_bytes();assert H(b)==st['source_sha256'];lines=b.decode(proof['sources'][path]['snippet_encoding']).splitlines();start=st['context_start_line']-1;ctx=st['context'].splitlines();assert lines[start:start+len(ctx)]==ctx;checks+=1
   for ev in op['updates']['ROI_membership_proof']['evidence']:
    dd=proof['sources'][ev['path']];ll=dd['snippet'].splitlines();assert all(ll[q['line']-1]==q['text']for q in ev['line_evidence']);checks+=len(ev['line_evidence'])
 for d in proof['scientific_intent'].values():assert H(Path(d['binding']['path']).read_bytes())==d['binding']['sha256'];checks+=1
 # Revalidate all32 selection records and exact source12 parent INDEX gzip joins.
 original=json.loads(gzip.decompress((O/'selected_review_records.json.gz').read_bytes()));ix=json.loads((O/'selection_parent_INDEX.json').read_bytes());cache={}
 for row in original:
  parent=Path(row['parent']);rb=parent.read_bytes();raw=gzip.decompress(rb);entry=next(c for c in ix['cohorts']if W/c['path']==parent);assert H(rb)==entry['compressed_sha256']and H(raw)==entry['raw_sha256'];r=json.loads(raw)['records'][row['index']];assert r==row['record'];b,m=r['benchmark'],r['model'];led=json.loads((S/'benchmarks'/f'{b}.json').read_bytes());ct=led['accepted_named_contract'];ms=led['models'][m];assert ms['profiling_status']=='profile_process_completed';cap=W/ms['profile_evidence_manifest'];plan=json.loads(gzip.decompress((cap.parent/'plan.json.gz').read_bytes()));assert plan['configuration_id']==ct['configuration_id']==r['configuration_id'];assert plan['reference_result_sha256']==ct['result_sha256']==r['reference_result']['sha256'];assert plan['raw_config_sha256']==ct['raw_config_sha256']==r['raw_config_sha256'];checks+=6
 for row in p['selected_capture_bindings']+p['withheld_models']:
  for key in ['ledger','manifest','plan','derivation','provenance']:
   if key in row:assert H(Path(row[key]['path']).read_bytes())==row[key]['sha256'];checks+=1
 for d in p['sources']:assert H(Path(d['path']).read_bytes())==d['sha256'];checks+=1
 assert(S/'roi_reviews/INDEX.json').read_bytes()==(O/'selection_parent_INDEX.json').read_bytes();checks+=1
 (O/'final_locked_revalidation.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source12_INDEX_and_legacy_stable':True,'all32_target_records_exact_equal_original_indices':len(original),'exactselected_CID_config_result_currentjoins':32,'fullsources_and_quote_lines_and_stage_contexts_equal':True,'selected_capture_hashes_stable':True,'checks':checks,'metadata_lock':'LOCK_SH','target_execution':False},indent=2)+'\n')
encodingpath='/home/imo/HeCBench_julia/src/scan2-cuda/main.cu';d=proof['sources'][encodingpath];bb=Path(encodingpath).read_bytes();assert H(bb)=='b4ebfd20e21181df297456e8d162de10ca5b13e1850b383b0df7b022e78aa177'and d['snippet_encoding']=='latin-1'and d['snippet'].encode('latin-1')==bb
lineproofs=[]
for row in p['patches']:
 for op in row['stage_patches']:
  for ev in op['updates']['ROI_membership_proof']['evidence']+op['updates']['ROI_membership_proof']['timed_closure_boundaries']:
   if ev['path']!=encodingpath:continue
   ll=bb.decode('latin-1').splitlines(keepends=True)
   for q in ev['line_evidence']:
    i=q['line']-1;raw=ll[i].encode('latin-1');start=sum(len(x.encode('latin-1'))for x in ll[:i]);assert ll[i].rstrip('\r\n')==q['text']and bb[start:start+len(raw)]==raw;lineproofs.append({'benchmark':row['record_key']['benchmark'],'model':row['record_key']['model'],'stage_id':op['stage_id'],'line':i+1,'start_byte':start,'end_byte':start+len(raw),'encoding':'latin-1','text':ll[i]})
(O/'reversible_encoding_proof.json').write_text(json.dumps({'status':'PROPOSED_EXACT_ROOTNOTE_ENCODING_APPROVAL_REQUIRED_BEFORE_MERGE','proposed_approved_reversible_source_encodings':{encodingpath:{'sha256':H(bb),'encoding':'latin-1'}},'fullsource':d,'all_source_line_snippets_byteencode_offset_proofs':lineproofs,'fullbyte_inverse_equal':True,'genericfallback':False,'replace_ignore_skipheader':False,'other_badUTF8_header_unquoted':json.loads((O/'unquoted_source_scope.json').read_bytes())},indent=2)+'\n')
corrections=[]
for row in p['patches']:
 for op in row['stage_patches']:
  u=op['updates']
  if u.get('regex_status','').startswith(('exact_source_ternary','exact_selected_CUDA_entry')):corrections.append({'record_key':row['record_key'],'stage_id':op['stage_id'],'expected_original_stage_canonical_sha256':op['expected_original_stage_canonical_sha256'],'new_regex':u['model_kernel_regex'],'basis':u['regex_status'],'exactcallsite':u['ROI_membership_proof']['exact_stage_callsite'],'evidence':u['ROI_membership_proof']['evidence']})
(O/'selector_correction_proof.json').write_text(json.dumps({'source_only':True,'no_genericdispatcher_identity_inferred':True,'corrections':corrections},indent=2)+'\n')
matchrows=[x for r in p['patches']for x in r['selected_observation_matching']];ops=[s for r in p['patches']for s in r['stage_patches']];counts={'models':len(p['patches']),'names':len({r['record_key']['benchmark']for r in p['patches']}),'updates':len(ops),'inclusions':sum(s['updates']['semantic_family_in_primary_ROI']for s in ops),'exclusions':sum(not s['updates']['semantic_family_in_primary_ROI']for s in ops),'saved_rows':len(matchrows),'matched_saved_rows':sum(x['source_family_inclusion_only']for x in matchrows),'unassigned_saved_rows':sum(not x['source_family_inclusion_only']for x in matchrows),'inclusion_alias_nulls':sum(s['updates']['semantic_family_in_primary_ROI']and s['updates'].get('model_kernel_regex','MISSING')is None for s in ops),'holds':len(p['withheld_models']),'unresolved_stages_unchanged':len(p['withheld_stages'])};(O/'review_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts));print('CHECKS',checks)
