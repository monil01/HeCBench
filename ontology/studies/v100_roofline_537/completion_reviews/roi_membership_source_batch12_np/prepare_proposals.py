import json,gzip,hashlib,re,datetime,fcntl,copy
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N='pitch pnpoly pns present prna projectile'.split()
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def file(p):p=Path(p);return {'path':str(p),'sha256':sha(p.read_bytes())}
def load(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def quoted(p):
 p=Path(p).resolve();b=p.read_bytes();encoding='utf-8'
 exact={'/home/imo/HeCBench_julia/src/projectile-cuda/Projectile.hpp':'c5ad1618143549433ec76937c5e21beb06dcc0af0ec046c0dd3b2e02fd9352bf'}
 if str(p)in exact:
  assert sha(b)==exact[str(p)];encoding='latin-1'
 try:t=b.decode(encoding)
 except UnicodeDecodeError:print('NON_UTF8_EXACT_SOURCE',str(p),sha(b));raise
 assert t.encode(encoding)==b
 return {**file(p),'snippet_encoding':encoding,'start_line':1,'end_line':t.count('\n')+1,'start_byte':0,'end_byte':len(b),'snippet':t}
why={'pitch': 'Selected Fixture::launch/launch! applies three in-place Float32 sigmoid steps across four 2D/3D shapes and pitched/simple modes; all eight path means from five reset-excluded internal batches summed. Original four device bodies and actual cudaMallocPitch/cudaMalloc3D row/slice strides retained, but named initialization repairs original undefined-input native reference; qualification/exclusion from untouched-native aggregates unchanged. Setup/warm/reset/input upload/full physical padding guards and Float64 mathematical oracle outside timer. Source dimensional/Val dispatch eligible, sampled mode/shape/step and full elapsed ROI unproved; no array-layout speed causality or network application claim.', 'pnpoly': 'Selected run/run! executes base plus seven point-tile modes 0,1,2,4,8,16,32,64 for a 600-vertex dyadic rectangle and16384 sample points. Float32 ray-crossing predicates produce exact Int32 membership; ten launch+completion mean per mode summed. Original native main uncalled; all seven pnpoly_opt template instances share existing family selector and are aliases, not duplicate counter attribution or sampled-template proof. Warm/setup/readback/analytic rectangle checks outside. No robust arbitrary geometry/boundary policy or application dataset certification.', 'pns': 'Selected factory launches one 256-thread block per four Petri-net trajectories, N1024 S312 seed5489..5492, stochastic token flow with UInt8 physical marks, Int32 conflicts, UInt32 MT state and Float32 variance reward. One launch+completion inside timer after zero-initialized conflict reset; input/full state/conflicts/token-mass/reward independent checks outside. Native same-line parent warm/timed selector roles held rather than guessed. Defined conflicts repair and bounded one624-attempt source schedule qualifications retained; no arbitrary source concurrency, long-time ensemble or real stochastic model validation.', 'present': 'Selected run/run! computes PRESENT-80 encryption for1025 independent blocks,31 full rounds and final whitening, UInt8 states/key schedule/table permutation, no scalar floating-point operator. Ten launches each followed by host synchronization averaged inside timer; setup/warm/readback/full8200byte independent spec/KAT comparisons outside. Native original renamed main uncalled. Source includes source tail guard; sampled block/round/phase unproved and no external security, throughput extrapolation or application system claim.', 'prna': 'Selected Case::pipeline/launch_pipeline executes boundary init plus eight stages per sequence diagonal over five RNA cases with Float32 negative-log partition recurrence, parameters, impossible +Inf states and pairing constraints. Full pipeline+context completion percase timer summed; source helper also called in untimed27-base209-snapshot diagnostic replay, so source family does not identify sampled invocation phase. Reset/prefix/capture, host probability evaluation/fullstate/tiny control checks and trace replay outside. Native original prna wrapper uncalled; Julia Val stage dispatch explicit, repeated partial-kernel source selectors aliases. Source310.15K thermodynamic normalization retained, unused zero temperature field is not0K; no experimentally validated folding/structure prediction, complete ROI or causal performance claim.', 'projectile': 'Selected CalculateRange and explicit Julia callback map1048576 deterministic angle/velocity pairs to range,time,height using Float32 sin/cos, source3.1415 and9.81 constants; ten launch loop+completion mean timed. Native separate warmup call excluded and timed call included by distinct source lines; CUDAjl unresolved parent selector held without new regex. Setup/reset/CPU algebra oracle/full output readback outside. Preserve nonstandard source height (v*v*s*s)/2*9.81, not standard physical-law validation; no trajectory integration, physical mechanics certification or sampled angle/phase/full ROI claim.'}
rows=json.loads(gzip.decompress((O/'selected_review_records.json.gz').read_bytes()));proposals=[];parents={};sources={};selected=[];withheld=[];withheld_models=[];snips={};scientific={}
with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 indexpath=S/'roi_reviews/INDEX.json';indexraw=indexpath.read_bytes();index=json.loads(indexraw);assert indexraw==(O/'selection_parent_INDEX.json').read_bytes(),'Canonical INDEX changed after selection; require explicit rebase'
 parent_index_checks=[]
 for row in rows:
  r=row['record'];b,m=r['benchmark'],r['model'];parent=Path(row['parent']);pb=parent.read_bytes();raw=gzip.decompress(pb);actual=json.loads(raw)['records'][row['index']];assert actual==r
  lp=S/'benchmarks'/f'{b}.json';ledger=json.loads(lp.read_text());state=ledger['models'][m];assert state['profiling_status']=='profile_process_completed'
  mp=W/state['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';op=mp.parent/'derived.json.gz';vp=mp.parent/'profile_provenance.json.gz';plan=load(pp);obs=load(op)['records'];prov=load(vp)
  contract=ledger['accepted_named_contract']
  if any(plan[pk]!=contract.get(ck)for pk,ck in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]):
   withheld_models.append({'benchmark':b,'model':m,'reason':'Exact canonical selected-contract CID/result/config join fails; no repair or relaxation.','ledger':file(lp),'plan':file(pp)});continue
  assert plan['configuration_id']==r['configuration_id'] and plan['reference_result_sha256']==r['reference_result']['sha256']and plan['raw_config_sha256']==r['raw_config_sha256']
  for key in ['reference_result','raw_config']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
  if b=='pns'and m=='cuda':
   assert r['stages'][0]['source_callsite']==r['stages'][1]['source_callsite']and r['stages'][0]['source_parameter_dispatch']==r['stages'][1]['source_parameter_dispatch']
   withheld_models.append({'benchmark':b,'model':m,'reason':'Selected native warm and timed launches have byte-identical stored line/context/expression selectors; exact parent stage-role correspondence is not independently encoded. Hold rather than assign ordinal or invent a stage.','ledger':file(lp),'manifest':file(mp),'plan':file(pp),'derivation':file(op),'provenance':file(vp),'original_record_canonical_sha256':canon(r)});continue
  bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()};bindings.update({str(Path(k).resolve()):v for k,v in prov.get('additional_current_profile_input_bindings',{}).items()})
  # Include all selected repository text bindings, retaining actual execution source instead of fabricating historical copies.
  paths={Path(p)for p in bindings if p.startswith(str(W/'src')+'/')and Path(p).suffix in ['.cu','.jl','.h','.hpp','.cuh','.inc','.cxx','.cpp','.c']}
  for s in r['stages']:
   if 'source_callsite'in s:paths.add(Path(s['source_callsite']['path']).resolve())
  ev=[]
  for p in sorted(paths):
   # AVI I/O is outside these kernel timers; retain exact captured bytes/hash
   # without making text quotations or source-callpath claims about its decoder.
   if p.name in ('avilib.c','avilib.h'):
    efile=file(p);assert bindings.get(str(p))==efile['sha256'];sources[str(p)]=efile;continue
   e=quoted(p);assert bindings.get(str(p))==e['sha256'],('captured source mismatch',b,m,str(p));sources[str(p)]=file(p)
   # Compact exact source spans in the proposal; full snippets deduplicated in compressed proof packet.
   lines=e['snippet'].splitlines();picked=[]
   pattern=r'launch|factory|pipeline|study\(|study_|time_ns|chrono::|main\(|#define main|#undef main|kernel|case [4-7]:|c\.k==|mode=='
   for i,line in enumerate(lines):
    if re.search(pattern,line):picked.append({'line':i+1,'text':line})
   ev.append({'path':str(p),'sha256':e['sha256'],'line_evidence':picked})
   snips[str(p)]=e
  domain=W/'ontology/studies/julia_abstractions_537/domains'/f'{b}.json';d=json.loads(domain.read_text());scientific[b]={'binding':file(domain),'scientific_intent':d.get('scientific_intent'),'native_selected_configuration_id':r['configuration_id'],'native_scientific_intent_record_link':r.get('scientific_intent_source')}
  patches=[];canonical_family={}
  for si,s in enumerate(r['stages']):
   if 'source_callsite'not in s:
    withheld.append({'benchmark':b,'model':m,'stage_id':s['stage_id'],'reason':'Existing stage has no exact source callsite/selector; no new stage or guessed identity supplied.'});continue
   path=Path(s['source_callsite']['path']).resolve();kind='timed_semantic_family';include=True
   # Distinct selected helper and archived CLI callpaths.
   if m=='cuda':
    limits={'pitch':4,'pnpoly':8,'present':1,'prna':9}
    if b in limits and si>=limits[b]:kind='uncalled_original_CLI_or_host_wrapper';include=False
    if b=='projectile'and si==0:kind='distinct_untimed_warmup_callsite';include=False
   u={'semantic_family_in_primary_ROI':include,'ROI_membership_status':'source_timed_closure_inclusion_proved'if include else 'source_setup_exclusion_proved'}
   if not include:u.update(model_kernel_regex=None,regex_status='disabled_source_excluded_'+kind+'_not_runtime_phase_proof')
   regex=u.get('model_kernel_regex',s.get('model_kernel_regex'))
   if include and regex:
    if regex in canonical_family:u.update(model_kernel_regex=None,alias_of_stage_id=canonical_family[regex],regex_status='same_family_source_callsite_alias_no_duplicate_counter_attribution')
    else:canonical_family[regex]=s['stage_id']
   u['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':ev,'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'scientific_intent_binding':scientific[b],'sample_in_primary_timed_invocation':None,'retained_trial':None,'whole_elapsed_ROI_claim':False,'complete_ROI':False,'causal_bottleneck':'unknown','limitations':['Source family inclusion/exclusion only; warmup/setup/tracing may execute same family; captured invocation phase unproved.','No whole-ROI completeness, launch multiplicity extrapolation, timing replacement, numerical recertification or causal bottleneck claim.','Generic callback/Val dispatch resolves family only; sampled argument/specialization is not inferred.']}
   patches.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canon(s),'updates':u})
  revised=copy.deepcopy(r['stages'])
  for opatch in patches:revised[opatch['stage_index']].update(opatch['updates'])
  matches=[]
  for ob in obs:
   hits=[st for st in revised if st.get('model_kernel_regex')and re.search(st['model_kernel_regex'],ob['identity']['Kernel Name'])];assert len(hits)<=1,(b,m,ob['identity']['Kernel Name'],[x['stage_id']for x in hits])
   matches.append({'identity':ob['identity'],'matched_stage_ids':[st['stage_id']for st in hits],'source_family_inclusion_only':bool(hits and hits[0].get('semantic_family_in_primary_ROI')),'runtime_first_sample_phase':'unproved'})
  if not any(x['source_family_inclusion_only']for x in matches):
   withheld_models.append({'benchmark':b,'model':m,'reason':'No positive source-included exact saved-family match; absent or vendor stages not invented, unknown signatures not broadened.','ledger':file(lp),'manifest':file(mp),'plan':file(pp),'derivation':file(op),'provenance':file(vp),'reviewed_but_not_submitted_stage_patches':patches});continue
  indexrows=[x for x in index['cohorts']if W/x['path']==parent];assert len(indexrows)==1
  assert indexrows[0]['compressed_sha256']==sha(pb)and indexrows[0]['raw_sha256']==sha(raw)
  assert json.loads(raw)['records'][row['index']]==r
  parent_index_checks.append({'benchmark':b,'model':m,'parent':str(parent),'record_index':row['index'],'record_canonical_sha256':canon(r),'INDEX_row':indexrows[0]})
  parents[str(parent)]={'compressed_sha256':sha(pb),'uncompressed_sha256':sha(raw)}
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':row['index'],'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canon(r),'stage_patches':patches,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'ledger':file(lp),'manifest':file(mp),'plan':file(pp),'derivation':file(op),'provenance':file(vp)})
 for p,e in sources.items():assert sha(Path(p).read_bytes())==e['sha256']
 for p,e in parents.items():assert sha(Path(p).read_bytes())==e['compressed_sha256']
 for r in selected:
  for k in ['ledger','manifest','plan','derivation','provenance']:assert sha(Path(r[k]['path']).read_bytes())==r[k]['sha256']
 assert indexpath.read_bytes()==indexraw,'Canonical INDEX changed during review'
 (O/'parent_INDEX_before_after.json').write_text(json.dumps({'INDEX_binding':file(indexpath),'INDEX_bytes_stable':True,'all_original_target_record_bytes_equal':True,'parent_record_INDEX_joins':parent_index_checks},indent=2)+'\n')
 out={'schema_version':1,'scope':'Private source-callpath membership batch12 six remaining eligible completed-all4 n then p names, both authorities and all batch11 proposals and holds excluded; root scientific review and merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_stages':withheld,'withheld_models':withheld_models,'merging_policy':'Recheck initial parent compressed/raw SHA, canonical record/stage SHA and selected/source hashes. Immutable historical bytes and all phase/ROI/causal qualifiers preserved; root merger population constants need explicit batch-specific adaptation.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
 (O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n')
 proof={'sources':snips,'scientific_intent':scientific,'reviewed_names':N,'capture_bindings':selected,'scope':'Deduplicated exact full source bytes supporting source-only decisions; not new execution evidence'}
 (O/'proofpacket.json.gz').write_bytes(gzip.compress((json.dumps(proof,indent=2)+'\n').encode(),mtime=0))
 summary={'names':N,'model_records':len(proposals),'withheld_models':withheld_models,'stage_updates':sum(len(r['stage_patches'])for r in proposals),'inclusions':sum(st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'exclusions':sum(not st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'withheld_vendor_placeholders':len(withheld),'sources':len(sources),'proposal_sha256':sha((O/'proposed_stage_patches.json').read_bytes()),'proofpacket_sha256':sha((O/'proofpacket.json.gz').read_bytes()),'metadata_read_lock':'LOCK_SH entire construction and stable final rehash','no_target_execution':True}
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='withheld_models'},indent=2));print('WITHHELD',[(x['benchmark'],x['model'])for x in withheld_models])
