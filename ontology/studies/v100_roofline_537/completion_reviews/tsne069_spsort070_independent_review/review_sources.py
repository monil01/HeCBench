from pathlib import Path
import json,gzip,hashlib,fcntl,re,copy
O=Path(__file__).parent;P=O.with_name('tsne069_spsort070_exact_recovery_proposal');W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';cache={};checks=[]
def H(p):
 p=Path(p);st=p.stat();k=(str(p),st.st_size,st.st_mtime_ns)
 if k not in cache:
  h=hashlib.sha256()
  with p.open('rb')as f:
   for b in iter(lambda:f.read(4<<20),b''):h.update(b)
  cache[k]=h.hexdigest()
 return cache[k]
def ck(n,v,d=None):
 checks.append({'check':n,'passed':bool(v),'detail':d})
 if not v:raise ValueError((n,d))
def load(p):p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def move(v,a,b):
 if isinstance(v,str):return v.replace(a,b)
 if isinstance(v,list):return [move(x,a,b)for x in v]
 if isinstance(v,dict):return {k:move(x,a,b)for k,x in v.items()}
 return v
ck('controller_exact',H(P/'controller.py')=='094e5048bb61dc609ed0bc69898709df17420675b78ddfe249c9adf5b12515d3');ck('review_exact',H(P/'launch_review.json')=='4d8556df9c05a9628f91ff7e1f81ac7c0465e5f4eeacfc6ce00bde307c3f51f8');review=load(P/'launch_review.json');proof=load(P/'tsne_exact_source_recovery_proof.json');jobs=review['jobs'];ck('exact_five_jobs',{j['job_id']for j in jobs}=={'tsne__ka'}|{'spsort__'+m for m in ['cuda','cuda_jl','ka','jacc']})
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for group in ['tool_bindings','packet_bindings']:
  for path,h in review[group].items():ck('frozen_'+group,H(path)==h,path)
 captured_source=load(proof['frozen_source_proof']['path']);manifest=load(proof['frozen_manifest']['path']);item=next(x for x in manifest['files']if x['original_name']=='source_proof.json');ck('frozen_sourceproof_manifest',H(proof['frozen_source_proof']['path'])==item['compressed_sha256']and hashlib.sha256(gzip.decompress(Path(proof['frozen_source_proof']['path']).read_bytes())).hexdigest()==item['uncompressed_sha256'])
 for q in proof['sources']:
  actual=next(x for x in captured_source['patches']if x['copy']==q['retained_capture_source']);original=Path(q['canonical_source']['path']).read_bytes();retained=Path(q['recovered_exact_bytes']['path']).read_bytes();fresh=Path(q['fresh_source']['path']).read_bytes();ck('original_canonical_SHA',H(q['canonical_source']['path'])==actual['source_sha256'],q['canonical_source']['path']);ck('retained_complete_text_SHA',retained==actual['text'].encode()and hashlib.sha256(retained).hexdigest()==actual['copy_sha256']==q['retained_capture_sha256']);ck('historical_scratch_not_recreated',not Path(q['retained_capture_source']).exists())
  # Validate old/new hunk slices independently, then rebuild entire file including all unchanged spans.
  lines=actual['diff'].splitlines(keepends=True);src=original.decode().splitlines(keepends=True);dst=retained.decode().splitlines(keepends=True);spos=dpos=0;reconstructed=[];i=0
  while i<len(lines):
   if not lines[i].startswith('@@ '):i+=1;continue
   mat=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[i]);ck('strict_hunk_header',bool(mat));sa=int(mat[1])-1;da=int(mat[3])-1;ck('unchanged_between_hunks',src[spos:sa]==dst[dpos:da]);reconstructed+=src[spos:sa];i+=1;left=[];right=[]
   while i<len(lines)and not lines[i].startswith('@@ '):
    line=lines[i];i+=1
    if line.startswith(('--- ','+++ ')):continue
    ck('known_diff_tag',line[:1]in [' ','+','-']);
    if line[:1]in [' ','-']:left.append(line[1:])
    if line[:1]in [' ','+']:right.append(line[1:])
   ck('old_hunk_exact',src[sa:sa+len(left)]==left);ck('new_hunk_exact',dst[da:da+len(right)]==right);reconstructed+=right;spos=sa+len(left);dpos=da+len(right)
  ck('unchanged_tail',src[spos:]==dst[dpos:]);reconstructed+=src[spos:];ck('wholefile_forward_inverse', ''.join(reconstructed).encode()==retained)
  before=proof['old_scratch'];after=proof['future_scratch'];ck('future_only_literal_prefix',fresh==retained.replace(before.encode(),after.encode())and fresh.replace(after.encode(),before.encode())==retained)
 for j in jobs:
  ck('no_issues',not j['issues'],j['job_id']);old=load(j['original_plan']['path']);new=load(j['prepared_plan']['path']);ck('prepared_plan_SHA',H(j['prepared_plan']['path'])==j['prepared_plan']['sha256']);ck('original_plan_SHA',H(j['original_plan']['path'])==j['original_plan']['sha256']);ledger=load(S/'benchmarks'/f"{j['benchmark']}.json");ct=ledger['accepted_named_contract'];state=ledger['models'][j['model']];ck('current_contract',all(ct[k]==v for k,v in j['expected_contract'].items()),j['job_id']);ck('current_latest_capture',state['latest_profile_attempt']['profile_id']==j['expected_latest_profile_id'],j['job_id']);ck('original_frozen_plan_equal',old==load(W/j['expected_latest_profile_id']/'plan.json.gz'))
  for path,h in j['readonly_bindings'].items():ck('readonly_binding',H(path)==h,path)
  restored=copy.deepcopy(new)
  for cp in j['source_copies']:
   ck('frozen_sourcecopy',H(cp['frozen_copy']['path'])==cp['future_sha256']);ck('future_source_inside_fresh_scratch',Path(cp['future_copy']).is_relative_to(j['scratch']));del restored['execution_bindings'][cp['future_copy']]
  restored['execution_bindings'].update(j['restored_original_binding_keys'])
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:restored[k]=move(restored[k],j['scratch'],j['old_scratch'])
  ck('whole_original_plan_inverse',restored==old,j['job_id']);ck('flat_batchroot',j['plan_root']=='/tmp/hecbench-roofline-v100-20261020/'+('batch069'if j['group']=='tsne'else'batch070'));ck('fresh_rawroot_uncreated',not Path(j['plan_root']).exists());ck('no_retain_and_fullprotocol','--retain-auxiliary'not in j['argv']and'--skip-qualified-selected'in j['argv']and new['environment_overrides']['CUDA_VISIBLE_DEVICES']=='6');ck('explicit_timeout',j['argv'][j['argv'].index('--timeout')+1]==('1800'if j['group']=='tsne'else'3600'))
 sel=load(P/'selectors/spsort_all4_exact.json');ck('selector_all4',set(sel['selection'])=={'cuda','cuda_jl','ka','jacc'})
 for j in jobs:
  if j['benchmark']!='spsort':continue
  prov=load(W/j['expected_latest_profile_id']/'profile_provenance.json.gz');parent=load(prov['selection_override']['path']);ck('selector_exact_individual',sel['selection'][j['model']]==parent['selection'][j['model']]);ck('selector_fixture_current',H(sel['fixture_description']['path'])==sel['fixture_description']['sha256']);logpath=W/j['expected_latest_profile_id']/'execution.log.gz';log=gzip.decompress(logpath.read_bytes());ck('old_log_provenance_SHA',hashlib.sha256(log).hexdigest()==prov['profile']['log_sha256']);tr=[json.loads(x[12:])for x in log.decode().splitlines()if x.startswith('STUDY_TRIAL ')];ck('exact_two_actual_passed_oldrows',len(tr)==2 and [t['trial']for t in tr]==[1,2]and all(t['correct']is True and t['checked_values']==484725 and t['mismatch_output_words']==0 for t in tr),j['job_id']);ck('old_walltimeout_retained',prov['profile']['stop_reason']=='wall_timeout'and prov['operational_deadline']['requested_seconds']==600)
 fcntl.flock(lock,fcntl.LOCK_UN)
report={'source_review_passed':True,'root_launch_admissible_subject_to_root_atomic_revalidation':True,'static_checks':len(checks),'checks':checks,'runtime_validation_performed':False,'GPU_build_tests_imports_sharededits':0,'scope':['OneTSNEKA069,all4SPSORT070; onecollectorcall/group, no retry','Exact3full054source byte streams+independent fullfile hunk unchangedranges+literal069prefix inverse','Allfive wholeplans restore exactly; sourcecount/threshold/timer/codec/guards unchanged','SPSORT old041 exactly2completepassedtrials/model; new5mandatory,3600explicit','Currentall5CID/config/result/latestcapture andsource/runtime/tooljoins stable underSH','Default freshscratch postmanifestcleanup only; binaryrecertification not claimed','Runtime safety remains conditional; no timedphase/fullROI/causal promotion']};(O/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'report_sha256':H(O/'independent_review.json')}))
