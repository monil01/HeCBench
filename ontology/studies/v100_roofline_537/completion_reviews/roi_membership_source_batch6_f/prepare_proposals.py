import json,gzip,hashlib,re,datetime,fcntl,copy
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N='floydwarshall fluidSim fma fpdc fresnel frna'.split()
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def file(p):p=Path(p);return {'path':str(p),'sha256':sha(p.read_bytes())}
def load(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def quoted(p):
 p=Path(p).resolve();b=p.read_bytes();t=b.decode();return {**file(p),'start_line':1,'end_line':t.count('\n')+1,'start_byte':0,'end_byte':len(b),'snippet':t}
why={'floydwarshall': 'Native selected study main builds two64node fixtures; first floydWarshallPass callsite warms up before trials, second loops all64 k values between start and synchronization, while renamed historical_main is uncalled. Julia study_main prepare returns explicit CUDA/KA/JACC floyd callback; complete k0:63 pass loop occurs within t0-to-fence region after distance/path reset. Input/oracle/Dijkstra/routes/readback are outside timing. UInt32 distance/intermediate updates are graph integer work, not scalar FLOPs. First captured same-family invocation may be warmup; no actual timed phase inferred.', 'fluidSim': 'Selected native run lambda resets both population buffers, then times eight direct lbm launches with pointer swaps and final fence; host wrapper in included kernels.cu is not invoked by this entrypoint. Julia study::case! similarly resets six buffers before t=time_ns, then invokes explicit CUDA lbm_kernel, KA lbm_ka or JACC lbm_jacc callback eight times with swaps before fence. Two fixtures sum source-defined step means; readback includes final and scratch distributions. Float64 D2Q9 operator has source boundary/obstacle semantics; same run executes warmup, so source family inclusion does not establish actual sample phase.', 'fma': 'Native selected launch lambda zeroes both outputs then directly dispatches basic and rowwise implicit_fma kernels; twenty launch repeats lie after timer start and before device fence. Native generic host wrappers in kernels.h are uncalled; selected source definitions are retained in study_original_fp32.cuh. Julia fma_study launch resets c1/c2 inside OwnedStudy timed callback, then invokes bound CUDA/KA/JACC pair. CUDAjl study strips trailing exit(main(ARGS)); its two explicit callsites both name implicit_fma_kernel!, an inherited basic-kernel alias for both outputs, so same-family selector is aliased without claiming rowwise mapping. Fills are inside timer, output/oracle outside. Float32 serialization alone is not instruction evidence; no tensor execution or captured phase inferred.', 'fpdc': 'Selected corrected-secondary native wrapper compiles without STUDY_ORIGINAL, invokes warmup callsite before trials, then separate repeated CompressionKernel callsite under timer for two fixtures times d1/2/3/32. Header lines containing example <<<>>> syntax are comments, not launches; original native main is uncalled. Julia factory explicitly binds compression callback, fpdc_study times three repeats and context fences per case; input copies/output sentinel resets/readback/checks outside timer. Source payload is UInt64-bit codec with UInt8 bytes/Int32 offsets, not floating-point arithmetic merely because benchmark name contains fp. Preserve corrected-reference qualification and raw failed-original diagnostic separately. No sampled d/phase or wholeROI claim.', 'fresnel': 'Selected native wrapper calls exact source kernel once for warmup before trials, then separate callsite repeats kernel between timer and fence; historical_main is renamed/uninvoked. Main.cu defines that exact kernel which invokes Fresnel_Sine_Integral; selected plan binds explicit native_reference_repaired.cu separately, preserving corrected auxiliary sine formula contract. Julia prepare explicitly binds named fresnel_kernel in CUDA/KA/JACC and study_main repeats closure inside t0-to-fence timing; corrected expected values and independent quadrature built before timing, readback afterward. Generic native source name kernel is resolved by its exact bound definition and callsite, not assigned from an unknown signature. No sampled branch, instruction mix, tensor or timing phase inferred.', 'frna': 'Selected native Case::init invokes init_w5_and_w3; Case::step invokes nine explicit recurrence kernels. Trial measure lambda times init and each distance step separately with context fences; full state captures/checks lie after each segment. Julia rna_study mirrors launch_init and launch_step under each timer; CUDA/KA/JACC model launchers explicitly bind all ten named stage callbacks and retain ordered recurrence path. Configured-native/original frna.cu host pipeline callsites are not called by selected study_native entrypoint, whose kernels come from extracted native_kernels.cuh. Initialization is included here because source timer explicitly includes it; no name-based exclusion. Dynamic-programming energies/sequence/state are Int32, Float64 artifact serialization does not promote compute precision. Same pipeline warms up before trials; no actual invocation-phase or biological77baseglobal-optimum claim.'}
rows=json.loads(gzip.decompress((O/'selected_review_records.json.gz').read_bytes()));proposals=[];parents={};sources={};selected=[];withheld=[];withheld_models=[];snips={};scientific={}
with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for row in rows:
  r=row['record'];b,m=r['benchmark'],r['model'];parent=Path(row['parent']);pb=parent.read_bytes();raw=gzip.decompress(pb);actual=json.loads(raw)['records'][row['index']];assert actual==r
  lp=S/'benchmarks'/f'{b}.json';ledger=json.loads(lp.read_text());state=ledger['models'][m];assert state['profiling_status']=='profile_process_completed'
  mp=W/state['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';op=mp.parent/'derived.json.gz';vp=mp.parent/'profile_provenance.json.gz';plan=load(pp);obs=load(op)['records'];prov=load(vp)
  contract=ledger['accepted_named_contract']
  if any(plan[pk]!=contract.get(ck)for pk,ck in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]):
   withheld_models.append({'benchmark':b,'model':m,'reason':'Exact canonical selected-contract CID/result/config join fails; no repair or relaxation.','ledger':file(lp),'plan':file(pp)});continue
  assert plan['configuration_id']==r['configuration_id'] and plan['reference_result_sha256']==r['reference_result']['sha256']and plan['raw_config_sha256']==r['raw_config_sha256']
  for key in ['reference_result','raw_config']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
  bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()};bindings.update({str(Path(k).resolve()):v for k,v in prov.get('additional_current_profile_input_bindings',{}).items()})
  # Include all selected repository text bindings, retaining actual execution source instead of fabricating historical copies.
  paths={Path(p)for p in bindings if p.startswith(str(W/'src')+'/')and Path(p).suffix in ['.cu','.jl','.h','.hpp','.cuh']}
  for s in r['stages']:
   if 'source_callsite'in s:paths.add(Path(s['source_callsite']['path']).resolve())
  ev=[]
  for p in sorted(paths):
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
    withheld.append({'benchmark':b,'model':m,'stage_id':s['stage_id'],'reason':'Vendor placeholder has no exact source callsite; no blanket membership or runtime phase inferred.'});continue
   path=Path(s['source_callsite']['path']).resolve();kind='timed_semantic_family';include=True
   # Preserve exact selected timer branches; a comment example is not executable.
   if m=='cuda':
    if b=='floydwarshall'and si!=1:kind='warmup_before_timer'if si==0 else 'dormant_legacy_entrypoint';include=False
    if b=='fluidSim'and si>=1:kind='uncalled_legacy_host_wrapper';include=False
    if b=='fma'and si>=2:kind='uncalled_legacy_host_wrapper';include=False
    if b=='fpdc'and si!=1:kind='warmup_before_timer'if si==0 else 'nonexecutable_comment_example'if si in [2,4,6] else 'uncalled_original_native_main';include=False
    if b=='fresnel'and si!=1:kind='warmup_before_timer'if si==0 else 'dormant_legacy_entrypoint';include=False
    if b=='frna'and si>=10:kind='uncalled_configured_or_original_host_pipeline';include=False
   u={'semantic_family_in_primary_ROI':include,'ROI_membership_status':'source_timed_closure_inclusion_proved'if include else 'source_setup_exclusion_proved'}
   if not include:u.update(model_kernel_regex=None,regex_status='disabled_source_excluded_'+kind+'_not_runtime_phase_proof')
   if include and b=='fma'and m=='cuda_jl':u.update(model_kernel_regex=r'\bimplicit_fma_kernel_(?=[(<])',regex_status='exact_explicit_source_callable_alias_not_runtime_phase_proof')
   if include and b=='fresnel'and m=='cuda':u.update(model_kernel_regex=r'\bkernel(?=[(<])',regex_status='exact_bound_source_kernel_definition_and_selected_callsite_not_unknown_signature')
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
   withheld_models.append({'benchmark':b,'model':m,'reason':'No positive source-included exact saved-family match; no signature guessing or selector broadening.'});continue
  parents[str(parent)]={'compressed_sha256':sha(pb),'uncompressed_sha256':sha(raw)}
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':row['index'],'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canon(r),'stage_patches':patches,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'ledger':file(lp),'manifest':file(mp),'plan':file(pp),'derivation':file(op),'provenance':file(vp)})
 for p,e in sources.items():assert sha(Path(p).read_bytes())==e['sha256']
 for p,e in parents.items():assert sha(Path(p).read_bytes())==e['compressed_sha256']
 for r in selected:
  for k in ['ledger','manifest','plan','derivation','provenance']:assert sha(Path(r[k]['path']).read_bytes())==r[k]['sha256']
 out={'schema_version':1,'scope':'Private source-callpath membership batch6 six alphabetically selected completed-all4 names after flip; root scientific review and merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_stages':withheld,'withheld_models':withheld_models,'merging_policy':'Recheck initial parent compressed/raw SHA, canonical record/stage SHA and selected/source hashes. Immutable historical bytes and all phase/ROI/causal qualifiers preserved; root merger population constants need explicit batch-specific adaptation.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
 (O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n')
 proof={'sources':snips,'scientific_intent':scientific,'reviewed_names':N,'capture_bindings':selected,'scope':'Deduplicated exact full source bytes supporting source-only decisions; not new execution evidence'}
 (O/'proofpacket.json.gz').write_bytes(gzip.compress((json.dumps(proof,indent=2)+'\n').encode(),mtime=0))
 summary={'names':N,'model_records':len(proposals),'withheld_models':withheld_models,'stage_updates':sum(len(r['stage_patches'])for r in proposals),'inclusions':sum(st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'exclusions':sum(not st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'withheld_vendor_placeholders':len(withheld),'sources':len(sources),'proposal_sha256':sha((O/'proposed_stage_patches.json').read_bytes()),'proofpacket_sha256':sha((O/'proofpacket.json.gz').read_bytes()),'metadata_read_lock':'LOCK_SH entire construction and stable final rehash','no_target_execution':True}
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
