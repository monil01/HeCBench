from pathlib import Path
O=Path(__file__).parent;s=Path('/tmp/roofline-completion-20261021/roi_membership_batch5_stencil/prepare_proposals.py').read_text();a=s.index('N=');z=s.index('\ndef sha',a);s=s[:a]+"N=set('gaussian gd ge-spmm geglu gelu glu'.split())"+s[z:]
why={
'gaussian':'Selected native launch loops t0..126 with direct fan1 then fan2 inside synchronized timer; reset H2D/zeroes before timer, host BackSub and residual/oracle after. Native original gaussianElim main renamed/uninvoked. Julia factory explicitly returns same two-kernel forward-elimination loop (CUDAjl named functions, KA handles, JACC callbacks); gaussian_study passes operation! to OwnedStudy.run_trials!, whose reset precedes timer and readback/backsolve follows completion. Source Float32 nonpivoted elimination n128; reported timer covers forward elimination, not full linear solve or CPU backsolve. Device sampling phase unknown.',
'gd':'Native train resets x before stamp then32iterations reset objective/l2/correct/gradient followed by compute and update inside synchronized timer. Included original main renamed/uninvoked. Julia training! has same zero-reset before time_ns and32iter fill! plus actual run! callback before sync/elapsed. Model wrappers explicitly bind compute and update; shared compute_body! has sparse Float32 dot/logistic exp/log and atomics; update includes weight decay and l2 reduction. CPU Float64 oracle and complete output readback outside. Memset/fill work IS inside timed training closure; unmatched fill kernel observations remain unassigned. This scoped sparse logistic-regression training state is not dataset prediction quality.',
'ge-spmm':'Native call(method) chooses spmm_test1/2/3/4 and selected study visits all4methods,100repeats per synchronized timer after output memset. Julia study_main visits methods1:4 and invokes study_prepare-returned closure within study_loop! timer; actual callbacks explicit CUDA spmm_kernel!, KA/JACC portable_spmm!. CUDAjl study removes trailing exit(main(ARGS)) so legacy main candidates dormant. Source Float32 CSR sparse-by-dense multiplication at nr257/nc263/bc256. Native kernels use four distinct coarsening/mapping methods, Julia uses method-controlled column loop; source family membership does not establish identical lowering, full source method equivalence or causal resource effect. Allocation/warmup/oracle outside.',
'geglu':'Native timed study repeats geglu_gpu wrapper, whose source calls explicitly templated geglu_kernel through dimension/forloop dispatch; wrapper is CALLED, unlike renamed historical main. Three supported widths1280/2560/5120,rows128 all within source timers. Julia study_main invokes prepare closure repeatedly then synchronize; explicit model callback geglu_kernel! binds Float32 erf gate calculation. Source x*gate*0.5*(1+erf(gate/sqrt2)); distinct from tanh-approx GELU and sigmoid GLU. Setup/warmup/Float64 scientific oracle/readback outside; scalar/SFU compiler instruction attribution still counter-specific.',
'gelu':'All four captured commands explicitly select vec2,10repeats. Native timed launch chooses gelu_bias_loop, not gelu_bias_loop_base. Source kernel definitions are in hash-bound study_original_kernels.cuh; original main.cu candidates not called. Julia study sets mode from ARGS, factory explicit branch selects vec2 callback, and OwnedStudy times repeated operation after reset; CUDAjl study strips legacy exit(main(ARGS)). Baseline branch is excluded for selectedconfiguration, not unsupported/uncompiled globally. Source storage Float16; rounded half bias sum, Float32 tanh-polynomial evaluation and half projection; artifact Float32 widening outside timer. Repeated in-place operator, not a full neural-network workflow.',
'glu':'Native study has first glu_kernel warmup before timer, second direct glu_kernel inside timed repeat, original historical main renamed/uninvoked. Julia study_main prepares seven source axis choices(2,0,1,2,0,1,2), invokes supplied closure in each synchronized repeat timer; explicit CUDA/KA/JACC glu_kernel! callback. Float32 x1*sigmoid(x2), dimension32 tensor, transfer/setup/warmup/oracle outside. Seven paths include repeated axis choices; aliases are not independent kernel families. Output scope is gated activation, not model quality.'}
a=s.index('why=');z=s.index('\nlock=',a);s=s[:a]+'why='+repr(why)+s[z:]
a=s.index('  overlays=[]');z=s.index('  ev=[];',a)
s=s[:a]+'''  overlays=[]
  assert not plan.get('capture_overlay_provenance',{}).get('patches'),('requires independent overlay review',b,m)
  if m=='cuda':
   paths.add(base/('study_native.cu'if b in ['gaussian','gd','gelu']else'study_cuda.cu'))
   if b=='gelu':paths.add(base/'study_original_kernels.cuh')
  else:
   paths.add(base/'study_common.jl')
   for arg in plan['profile_target_command']:
    if str(arg).endswith('.jl')and Path(arg).is_file():paths.add(Path(arg).resolve())
   if b in ['gaussian','gelu']:paths.add(W/'src/_portable_study/OwnedStudy.jl')
'''+s[z:]
a=s.index("   path=Path(s['source_callsite']['path']);legacy=");z=s.index("   updates['ROI_membership_proof']=",a)
s=s[:a]+'''   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and legacy and b!='geglu')or(m=='cuda_jl'and b in ['ge-spmm','gelu']and path.name=='main.jl')
   warmup=(m=='cuda'and b=='glu'and si==0)
   unselected=False
   if b=='gelu':
    assert plan['profile_target_command'][-2:]==['vec2','10']
    unselected=(m=='cuda'and si==1)or(m in ['cuda_jl','ka']and si==0)
   kind='dormant_legacy_entrypoint'if dormant else'warmup'if warmup else'unselected_baseline_mode_branch'if unselected else'timed_semantic_family'
   included=not(dormant or warmup or unselected)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_warmup_or_dormant_alias_or_captured_argv_unselected_mode')
   if b=='gelu'and m=='jacc':
    updates.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\\n]*\\bgelu_vec2_jacc_(?=[,<(])',resolved_callable_candidates=['gelu_vec2_jacc!'],regex_status='exact_captured_vec2_mode_source_ternary_binding_baseline_not_selected')
'''+s[z:]
s=s.replace('Private batch5 source-only enrichment ten named cases','Private batch6 source-only enrichment six named cases').replace('Private batch5 source-only enrichment','Private batch6 source-only enrichment')
(O/'prepare_proposals.py').write_text(s)
