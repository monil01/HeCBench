from pathlib import Path
O=Path(__file__).parent;s=Path('/tmp/roofline-completion-20261021/roi_membership_batch7_stencil/prepare_proposals.py').read_text();a=s.index('N=');z=s.index('\ndef sha',a);s=s[:a]+"N=set('kiss kmeans knn kurtosis langevin'.split())"+s[z:]
why={
'kiss':'Native selected run(Case) calls uniform_distribution<UInt64,kiss::Kiss<UInt64>>, whose originalsource helper directly launches kernel with fixed4096blocks×256threads; that wrapper is called despite originalmain renamed. Julia supplied launch explicitly binds rng_kernel! and kiss_body!; shared study calls it within percase repeated timer, for3exact streamlength/seed fixtures. Guardreset/fill and warmup/readback/oracle outside. Source integer KISS64 modular pseudorandom stream generation; no scalarFP or MonteCarlo endpoint/statisticalquality claim from exact output sequence. Saved fill kernels remain unassigned.',
'kmeans':'Native selected study directly launches transpose and membership for warmup before trials (first2aliases excluded), then times transpose per k and iteration centroidH2D/membership/D2H/CPUupdate, plus separately timed hostRMSE. Originalcluster.cu wrapper launch candidates are not called; study owns matched source-shaped host controller. Julia study(alloc,first,second) invokes explicit first/second callbacks in corresponding timer segments; CUDAjl first/second hostnames resolve to transpose_kernel!/membership_kernel!, KAhandles and JACCcallbacks explicit. Source Float32 Kmeans across3named point/feature/cluster/restart fixtures, preserving integer-truncated RMSE selection. Allocation/reset/inputecho/oracle/trace writes outside; not full elapsed walltime or dataset-quality optimality guarantee.',
'knn':'Selected native F.call invokes original knn_parallel wrapper, whose three direct kernels compute distances, insertion-sort neighbors and sqrt. That wrapper is called inside10repeat twofixturetimer; its allocation/H2D/D2H/free is inside fullcall timer, not dormant originalmain. Julia supplied knn_call! allocates/uploads, dispatches named threecallbacks, copies distances+indices and frees before timedclosure returns. KA/JACC stage wrappers explicit; source Float32 distance arithmetic and integer stable tie indices, CPUFloat64 sorted oracle/readbackchecks aftertimer. No kernel-counter to fullhostelapsed-rate join; source model only exact neighbors on named fixtures, not downstream classification accuracy.',
'kurtosis':'Julia kurtosis_study operation allocates partial/output KResult arrays, calls supplied launch! twice (Val0 loads KStore,Val1 merges KResult), downloads one result and frees buffers, repeated10times in each7case timer. CUDAjl and JACC callsite selectors distinguish actual Val0/Val1; KA namedhandle explicitly invoked twice. Source count/mean and UNNORMALIZED central sums m2/m3/m4, not normalized/excess kurtosis statistic. Input AoS24 with Float32value and Float32moment merge; Float64artifact encoding outsideclock. Source independent fullfield budgets/readback aftertimers; warmup separate. Native unresolvedvendorstage held; no emittedThrust/CUBkernel attribution fromAPI alone.',
'langevin':'Native selected launch(mode) calls k0/k1/k2 inside percase20repeat timers, originalmain duplicatecalls dormant. Julia langevin_study invokes factory-returned modecallback for mode1:3 in sixcase timers; CUDAjl active factory callsites are afterlegacy main definition (legacyloopsnotcalled), KA tupleexplicit namedhandles, JACC tuplekernels=(portable0!,portable1!,portable2!) selectedby samecomplete mode1:3loop. Source Float32 coth(x)-1/x via cosh/sinh, reciprocal tanh, and explicit FMApolynomial; not Langevin stochastic trajectory. Sourcefidelity/mathematicalfunctionaccuracy policies distinct, reset/setup/warmup/oracle/guards outsideclock; samefamily sampling phaseunknown.'}
a=s.index('why=');z=s.index('\nlock=',a);s=s[:a]+'why='+repr(why)+s[z:]
a=s.index('proposals=[];sources={}');z=s.index('\nfor parent',a);s=s[:a]+s[a:z]+'''
legacy_path=S/'ROI_REVIEW.json.gz';legacy_packed=legacy_path.read_bytes();legacy_raw=gzip.decompress(legacy_packed);legacy_doc=json.loads(legacy_raw);legacy_receipt_path=S/'ROI_REVIEW_RECEIPT.json';legacy_receipt=json.loads(legacy_receipt_path.read_bytes())
assert sha(legacy_packed)==legacy_receipt['sha256'] and sha(legacy_raw)==legacy_receipt['uncompressed_sha256']
assert not any(r['benchmark']in N for r in legacy_doc['records']), 'Legacy authority overlaps proposed names'
legacy_authority={'path':str(legacy_path),'compressed_sha256':sha(legacy_packed),'uncompressed_sha256':sha(legacy_raw),'receipt':file(legacy_receipt_path),'none_of_selected_names_present':True}
'''+s[z:]
a=s.index("  if b=='goulash'and m=='ka':");z=s.index('  ev=[];',a)
s=s[:a]+'''  assert not any(str(stage.get('ROI_membership_status','')).startswith('source_')for stage in r['stages']),('Already source reviewed',b,m)
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
'''+s[z:]
a=s.index("   path=Path(s['source_callsite']['path']);legacy=");z=s.index("   updates['ROI_membership_proof']=",a)
s=s[:a]+'''   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and legacy and b not in ['kiss','knn'])or(b=='langevin'and m=='cuda_jl'and si<3)
   warmup=(m=='cuda'and b=='kmeans'and si<2)
   kind='dormant_legacy_entrypoint_or_uncalled_wrapper'if dormant else'warmup'if warmup else'timed_semantic_family'
   included=not(dormant or warmup)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_warmup_or_uninvoked_legacy_controller_alias')
   if b=='kmeans'and m=='cuda_jl':
    callback=['transpose_kernel!','membership_kernel!'][si]
    updates.update(model_kernel_regex=r'\\b'+callback.replace('!','_')+r'(?=[(<])',resolved_callable_candidates=[callback],regex_status='exact_source_first_second_closure_to_actual_named_kernel')
   if b=='kurtosis'and m in ['cuda_jl','jacc']:
    updates.update(model_kernel_regex=s['model_kernel_regex']+r'[^\\n]*\\bVal<\\(int\\)'+str(si)+r'>',regex_status='exact_source_Val0_KStore_load_vs_Val1_KResult_merge_specialization')
   if b=='langevin'and m=='jacc':
    updates.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\\n]*\\bportable[012]_(?=[,<(])',resolved_callable_candidates=['portable0!','portable1!','portable2!'],regex_status='exact_source_callback_tuple_mode1to3_allvisited_in_timedloop_no_genericdispatcher_inference')
'''+s[z:]
s=s.replace("out={'schema_version':1,","out={'legacy_authority_exclusion_check':legacy_authority,'schema_version':1,")
s=s.replace('Private batch7 source-only enrichment five named cases','Private batch8 source-only enrichment five named cases').replace('Private batch7 source-only enrichment','Private batch8 source-only enrichment')
# Revalidate legacy authority with the same read-only locked end check.
s=s.replace("# Before/after bindings establish", "assert sha(legacy_path.read_bytes())==legacy_authority['compressed_sha256']\nassert sha(legacy_receipt_path.read_bytes())==legacy_authority['receipt']['sha256']\n# Before/after bindings establish")
(O/'prepare_proposals.py').write_text(s)
