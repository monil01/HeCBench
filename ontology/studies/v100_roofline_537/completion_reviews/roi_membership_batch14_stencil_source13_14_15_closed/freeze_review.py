from pathlib import Path
import json,gzip,hashlib,fcntl,re
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).parent;H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();canon=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest();d=json.loads((O/'proposed_stage_patches.json').read_bytes());checks=[];held=[]
def ck(n,v,detail=None):
 checks.append({'check':n,'passed':bool(v),'detail':detail})
 if not v:raise ValueError((n,detail))
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);ck('INDEX_current',H(d['parent_INDEX']['path'])==d['parent_INDEX']['sha256']);ck('legacy_current',H(d['legacy_authority_exclusion_check']['path'])==d['legacy_authority_exclusion_check']['sha256']);idx=json.loads(Path(d['parent_INDEX']['path']).read_bytes());docs={}
 for path,h in d['parent_reviews'].items():
  raw=Path(path).read_bytes();un=gzip.decompress(raw);ck('parent_gzip',hashlib.sha256(raw).hexdigest()==h['compressed_sha256']);ck('parent_raw',hashlib.sha256(un).hexdigest()==h['uncompressed_sha256']);docs[path]=json.loads(un);i=next(x for x in idx['cohorts']if str(W/x['path'])==path);ck('INDEX_parent_row',i['compressed_sha256']==h['compressed_sha256']and i['raw_sha256']==h['uncompressed_sha256'])
 for r in d['selected_capture_bindings']:
  for k in ['ledger','manifest','plan','derivation','provenance']:ck('capture_binding_current',H(r[k]['path'])==r[k]['sha256'],[r['benchmark'],r['model'],k])
 for e in d['sources']:ck('source_current',H(e['path'])==e['sha256'],e['path'])
 quotechecks=0
 for r in d['patches']:
  old=docs[r['parent_review']['path']]['records'][r['record_index']];ck('record_canonical',canon(old)==r['expected_original_record_canonical_sha256']);ck('record_key',all(old[k]==v for k,v in r['record_key'].items()));new=[dict(s)for s in old['stages']]
  for x in r['stage_patches']:
   st=old['stages'][x['stage_index']];ck('stage_canonical',canon(st)==x['expected_original_stage_canonical_sha256']and st['stage_id']==x['stage_id']);u=x['updates'];pr=u['ROI_membership_proof'];ck('original_callsite_exact',pr['exact_stage_callsite']==st['source_callsite']);ck('no_phase_promotion',pr['sample_in_primary_timed_invocation']is None and pr['retained_trial']is None and pr['complete_ROI']is False and pr['whole_elapsed_ROI_claim']is False and pr['causal_bottleneck']=='unknown')
   for e in pr['evidence']:
    ck('proof_source_SHA',H(e['path'])==e['sha256']);ls=Path(e['path']).read_text(encoding='utf-8').splitlines()
    for line in e['line_evidence']:ck('source_quote_exact',ls[line['line']-1]==line['text']);quotechecks+=1
   new[x['stage_index']].update(u)
  cap=next(v for v in d['selected_capture_bindings']if v['benchmark']==old['benchmark']and v['model']==old['model']);obs=json.loads(gzip.decompress(Path(cap['derivation']['path']).read_bytes()))['records'];actual=[]
  for ob in obs:
   hits=[s['stage_id']for s in new if s.get('semantic_family_in_primary_ROI')is True and s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],ob['identity']['Kernel Name'])];ck('unique_savedfamily',len(hits)<=1);actual.append((ob['identity'],hits))
  ck('positive_savedfamily',any(v for _,v in actual));ck('savedmatch_exact',actual==[(x['identity'],x['matched_stage_ids'])for x in r['selected_observation_matching']])
 for h in d['withheld_models']:
  b,m=h['benchmark'],h['model'];lp=S/'benchmarks'/f'{b}.json';l=json.loads(lp.read_bytes());mp=W/l['models'][m]['profile_evidence_manifest'];files=[lp,mp,mp.parent/'plan.json.gz',mp.parent/'profile_provenance.json.gz'];held.append({'benchmark':b,'model':m,'bindings':[{'path':str(p),'sha256':H(p)}for p in files],'current_named_contract_configuration_id':l['accepted_named_contract']['configuration_id'],'reason':h['reason']})
 fcntl.flock(lock,fcntl.LOCK_UN)
(O/'final_locked_revalidation.json').write_text(json.dumps({'status':'All declared admitted source/parent/capture/INDEX/legacy/quote/savedmatch bindings unchanged underSH','checks':len(checks),'quote_checks':quotechecks,'all_pass':True,'held_capture_bindings':held,'GPU_build_tests_imports':0},indent=2)+'\n');d['held_capture_bindings']=held;(O/'proposed_stage_patches.json').write_text(json.dumps(d,indent=2)+'\n');aliases=[{'benchmark':r['record_key']['benchmark'],'model':r['record_key']['model'],'stage_id':x['stage_id'],'alias':x['updates']['alias_of_stage_id']}for r in d['patches']for x in r['stage_patches']if 'alias_of_stage_id'in x['updates']];(O/'selector_and_alias_changes.json').write_text(json.dumps({'included_samefamily_aliases':aliases,'new_regex_or_callable_changes':[],'source_only_excluded_selectors_disabled':True,'no_new_stages':True},indent=2)+'\n');print(json.dumps({'checks':len(checks),'quotes':quotechecks,'aliases':len(aliases),'proposal_SHA':H(O/'proposed_stage_patches.json')}))
