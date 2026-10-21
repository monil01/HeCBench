import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set('kiss kmeans knn kurtosis langevin'.split())
def sha(x):return hashlib.sha256(x).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def evidence(p):
 b=p.read_bytes()
 try:t=b.decode();offset=0
 except UnicodeDecodeError:
  offset=0
  while True:
   try:t=b[offset:].decode();break
   except UnicodeDecodeError as ex:offset=b.index(b'\n',offset+ex.start)+1
 return {**file(p),'start_line':b[:offset].count(b'\n')+1,'end_line':b.count(b'\n')+1,'start_byte':offset,'end_byte':len(b),'snippet':t}
why={'kiss': 'Native selected run(Case) calls uniform_distribution<UInt64,kiss::Kiss<UInt64>>, whose originalsource helper directly launches kernel with fixed4096blocks×256threads; that wrapper is called despite originalmain renamed. Julia supplied launch explicitly binds rng_kernel! and kiss_body!; shared study calls it within percase repeated timer, for3exact streamlength/seed fixtures. Guardreset/fill and warmup/readback/oracle outside. Source integer KISS64 modular pseudorandom stream generation; no scalarFP or MonteCarlo endpoint/statisticalquality claim from exact output sequence. Saved fill kernels remain unassigned.', 'kmeans': 'Native selected study directly launches transpose and membership for warmup before trials (first2aliases excluded), then times transpose per k and iteration centroidH2D/membership/D2H/CPUupdate, plus separately timed hostRMSE. Originalcluster.cu wrapper launch candidates are not called; study owns matched source-shaped host controller. Julia study(alloc,first,second) invokes explicit first/second callbacks in corresponding timer segments; CUDAjl first/second hostnames resolve to transpose_kernel!/membership_kernel!, KAhandles and JACCcallbacks explicit. Source Float32 Kmeans across3named point/feature/cluster/restart fixtures, preserving integer-truncated RMSE selection. Allocation/reset/inputecho/oracle/trace writes outside; not full elapsed walltime or dataset-quality optimality guarantee.', 'knn': 'Selected native F.call invokes original knn_parallel wrapper, whose three direct kernels compute distances, insertion-sort neighbors and sqrt. That wrapper is called inside10repeat twofixturetimer; its allocation/H2D/D2H/free is inside fullcall timer, not dormant originalmain. Julia supplied knn_call! allocates/uploads, dispatches named threecallbacks, copies distances+indices and frees before timedclosure returns. KA/JACC stage wrappers explicit; source Float32 distance arithmetic and integer stable tie indices, CPUFloat64 sorted oracle/readbackchecks aftertimer. No kernel-counter to fullhostelapsed-rate join; source model only exact neighbors on named fixtures, not downstream classification accuracy.', 'kurtosis': 'Julia kurtosis_study operation allocates partial/output KResult arrays, calls supplied launch! twice (Val0 loads KStore,Val1 merges KResult), downloads one result and frees buffers, repeated10times in each7case timer. CUDAjl and JACC callsite selectors distinguish actual Val0/Val1; KA namedhandle explicitly invoked twice. Source count/mean and UNNORMALIZED central sums m2/m3/m4, not normalized/excess kurtosis statistic. Input AoS24 with Float32value and Float32moment merge; Float64artifact encoding outsideclock. Source independent fullfield budgets/readback aftertimers; warmup separate. Native unresolvedvendorstage held; no emittedThrust/CUBkernel attribution fromAPI alone.', 'langevin': 'Native selected launch(mode) calls k0/k1/k2 inside percase20repeat timers, originalmain duplicatecalls dormant. Julia langevin_study invokes factory-returned modecallback for mode1:3 in sixcase timers; CUDAjl active factory callsites are afterlegacy main definition (legacyloopsnotcalled), KA tupleexplicit namedhandles, JACC tuplekernels=(portable0!,portable1!,portable2!) selectedby samecomplete mode1:3loop. Source Float32 coth(x)-1/x via cosh/sinh, reciprocal tanh, and explicit FMApolynomial; not Langevin stochastic trajectory. Sourcefidelity/mathematicalfunctionaccuracy policies distinct, reset/setup/warmup/oracle/guards outsideclock; samefamily sampling phaseunknown.'}
lock=Path('/tmp/hecbench-roofline-metadata.lock').open('r');fcntl.flock(lock,fcntl.LOCK_SH)
proposals=[];sources={};parents={};selected=[];withheld=[]
legacy_path=S/'ROI_REVIEW.json.gz';legacy_packed=legacy_path.read_bytes();legacy_raw=gzip.decompress(legacy_packed);legacy_doc=json.loads(legacy_raw);legacy_receipt_path=S/'ROI_REVIEW_RECEIPT.json';legacy_receipt=json.loads(legacy_receipt_path.read_bytes())
assert sha(legacy_packed)==legacy_receipt['sha256'] and sha(legacy_raw)==legacy_receipt['uncompressed_sha256']
assert not any(r['benchmark']in N for r in legacy_doc['records']), 'Legacy authority overlaps proposed names'
legacy_authority={'path':str(legacy_path),'compressed_sha256':sha(legacy_packed),'uncompressed_sha256':sha(legacy_raw),'receipt':file(legacy_receipt_path),'none_of_selected_names_present':True}

for parent in sorted((S/'roi_reviews').glob('*.json.gz')):
 pb=parent.read_bytes();raw=gzip.decompress(pb);doc=json.loads(raw)
 for ri,r in enumerate(doc['records']):
  b,m=r['benchmark'],r['model']
  if b not in N:continue
  lp=S/'benchmarks'/f'{b}.json';lb=lp.read_bytes();state=json.loads(lb)['models'][m]
  if state['profiling_status']!='profile_process_completed':withheld.append({'benchmark':b,'model':m,'reason':state['profiling_status']});continue
  mp=W/state['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';op=mp.parent/'derived.json.gz';plan=load(pp);obs=load(op)['records']
  assert plan['configuration_id']==r['configuration_id'] and plan['reference_result_sha256']==r['reference_result']['sha256'] and plan['raw_config_sha256']==r['raw_config_sha256']
  for k in ['reference_result','raw_config']:assert sha(Path(r[k]['path']).read_bytes())==r[k]['sha256']
  base=W/'src'/f'{b}-julia'; paths={Path(s['source_callsite']['path']).resolve() for s in r['stages'] if s.get('source_callsite')}
  assert not any(str(stage.get('ROI_membership_status','')).startswith('source_')for stage in r['stages']),('Already source reviewed',b,m)
  if b=='kurtosis'and m=='cuda':
   source=base/'study_native.cu';assert plan['execution_bindings'].get(str(source))==file(source)['sha256']
   withheld.append({'benchmark':b,'model':m,'reason':'Existing indexed stage unresolved: no direct source_callsite or positive selector. ActualCUBtyped kernels retained but ThrustAPI call alone not promoted to specific emittedvendorfamily. Native numerical result and profiledcounters unchanged.','actual_saved_identities':[x['identity']for x in obs],'selected_config':r['raw_config'],'selected_result':r['reference_result'],'selected_plan':file(pp),'selected_manifest':file(mp),'selected_derivation':file(op),'pinned_source_evidence':evidence(source)});continue
  overlays=[]
  assert not plan.get('capture_overlay_provenance',{}).get('patches'),('requires independent overlay review',b,m)
  if m=='cuda':paths.add(base/('study_cuda.cu'if b=='knn'else'study_native.cu'))
  else:
   paths.add(base/'study_common.jl')
   for arg in plan['profile_target_command']:
    if str(arg).endswith('.jl')and Path(arg).is_file():paths.add(Path(arg).resolve())
   if b=='kiss':paths.add(base/'study_device.jl')
   if b=='kurtosis':paths.add(base/'study_math.jl')
  ev=[];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for p in sorted(paths):
   e=evidence(p)
   assert bindings.get(str(p.resolve()))==e['sha256'],('selected source binding gap',b,m,str(p))
   ev.append(e);sources[e['path']]=file(p)
  patches=[]
  for si,s in enumerate(r['stages']):
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and legacy and b not in ['kiss','knn'])or(b=='langevin'and m=='cuda_jl'and si<3)
   warmup=(m=='cuda'and b=='kmeans'and si<2)
   kind='dormant_legacy_entrypoint_or_uncalled_wrapper'if dormant else'warmup'if warmup else'timed_semantic_family'
   included=not(dormant or warmup)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_warmup_or_uninvoked_legacy_controller_alias')
   if b=='kmeans'and m=='cuda_jl':
    callback=['transpose_kernel!','membership_kernel!'][si]
    updates.update(model_kernel_regex=r'\b'+callback.replace('!','_')+r'(?=[(<])',resolved_callable_candidates=[callback],regex_status='exact_source_first_second_closure_to_actual_named_kernel')
   if b=='kurtosis'and m in ['cuda_jl','jacc']:
    updates.update(model_kernel_regex=s['model_kernel_regex']+r'[^\n]*\bVal<\(int\)'+str(si)+r'>',regex_status='exact_source_Val0_KStore_load_vs_Val1_KResult_merge_specialization')
   if b=='langevin'and m=='jacc':
    updates.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\n]*\bportable[012]_(?=[,<(])',resolved_callable_candidates=['portable0!','portable1!','portable2!'],regex_status='exact_source_callback_tuple_mode1to3_allvisited_in_timedloop_no_genericdispatcher_inference')
   updates['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':[{'path':e['path'],'sha256':e['sha256'],'timer_lines':[{'line':e['start_line']+i,'text':line}for i,line in enumerate(e['snippet'].splitlines())if any(v in line for v in ['time_ns()', 'steady_clock::now()', 'Clock::now()', 'excluded+=', 'run_trials!(', 'bm_study(factory)', 'exit(study_main(', 'bezier_study(', 'box_study(factory)'])]}for e in ev],'captured_overlay_source_evidence':overlays,'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'selected_environment_overrides':plan['environment_overrides'],'precision_scope':'Source types/operations/Val branches are in exact evidence; selectedenvironment retained. Float32/Float64/mixed integer/conversion/SFU/clock semantics are distinct from actual counter instruction class; no arithmetic mode inferred from benchmark name.','limitations':['Source inclusion only: first captured invocation may be warmup/setup; sample phase remains unproved.','No complete elapsed-ROI or complete launch census, multiplicity extrapolation, numerical recertification or causal bottleneck claim.','Alias nulling avoids repeated same-family candidates; it does not remove source calls or hide conditional branches.']}
   patches.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canon(s),'updates':updates})
  revised=[dict(s,**patches[i]['updates'])for i,s in enumerate(r['stages'])];matches=[]
  for ob in obs:
   name=ob['identity']['Kernel Name'];hits=[s['stage_id']for s in revised if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],name)]
   assert len(hits)<=1,(b,m,name,hits)
   matches.append({'identity':ob['identity'],'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  if not any(x['matched_stage_ids']for x in matches):
   withheld.append({'benchmark':b,'model':m,'reason':'No positive saved included-family match; proposed model withheld','selected_observations':matches});continue
  parents[str(parent)]={'compressed_sha256':sha(pb),'uncompressed_sha256':sha(raw)}
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canon(r),'stage_patches':patches,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'ledger':{'path':str(lp),'sha256':sha(lb)},'manifest':file(mp),'plan':file(pp),'derivation':file(op)})
assert sha(legacy_path.read_bytes())==legacy_authority['compressed_sha256']
assert sha(legacy_receipt_path.read_bytes())==legacy_authority['receipt']['sha256']
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'legacy_authority_exclusion_check':legacy_authority,'schema_version':1,'scope':'Private batch8 source-only enrichment five named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
