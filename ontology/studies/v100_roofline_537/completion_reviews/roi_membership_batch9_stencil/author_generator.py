from pathlib import Path
O=Path(__file__).parent;s=Path('/tmp/roofline-completion-20261021/roi_membership_batch8_stencil/prepare_proposals.py').read_text();a=s.index('N=');z=s.index('\ndef sha',a);s=s[:a]+"N=set('lavaMD layernorm layout lfib4 log2 logprob'.split())"+s[z:]
why={
'lavaMD':'Native selected run(Case) directly invokes md per threecases within synchronized singlelaunch timer after resetzero; originalmain launch candidate notcalled. Julia study invokes explicit launch(meta,rv,q,fv,alpha,nb) between time_ns and synchronize; CUDAjl/JACC md_kernel! delegates md_body!, KA namedkernel includes same tiledpairloop. Source Float32 Gaussian pairinteraction/forces, packed box/neighbor metadata and StudyFour storage,128threads/block and shared900Float32 tile; outputs accumulatedfromzero. Guard/inputecho/weightedforcebalance/oraclechecks aftertimer, allocation/initialwarmup before. Namedsingleforceoperator cases, not a timeintegrated molecular trajectory; serialized Float64 is encoding not compute precision.',
'layernorm':'Native selected study lambda invokes layernorm_forward dispatcher inside10repeat timer for6fixtures×3algorithms×6blockwidths; originalsource wrappers0/1/2 each invoke namedkernel then device_synchronize. Kernel0 source C%4 branch chooses templateUNROLL4 or1, exact selectors disambiguated. Julia layernorm_study builds explicit callback peralgorithm/blockwidth and calls10times with synchronize aftereach within timer; algorithm0 uses ln0_kernel!,1/2 use lnwarp_kernel! with Valmode. Factory bindings explicit forCUDAjl launchfalse handles, KA namedhandles, JACC callbacks. Source Float32 mean/variance/reciprocalstd affine normalization; vector4load mapping distinct from arithmeticprecision. Guardreset/warmup/inputecho/fulloracle after/beforetimers; no fullnetworktraining claim.',
'layout':'Captured all4commands explicitly soa,10. Native selected launch invokes SoAKernel<TREE_SIZE> inside10repeat timer; AoSbranch unselected and originalmain loops dormant. Julia layout_study selects soa fixture, factory explicit branch binds soa_kernel!/soa_ka!/soa_jacc!, then OwnedStudy times10launches after outputreset; CUDAjl study sets includeonly guard before including oldmain. Source 4096rows×4096integerentries, UInt32wrap sums projectedInt32; not scalarFPwork. No AoS/SoA performance comparison inferred from this oneconfiguration. Setup/transfers/oracle outside, firstsamplephase unknown.',
'lfib4':'Native timed pipeline launches firstColGPU and colYGPU on two streams, joins both, then launches lastEntGPU and colsGPU followed by devicecompletion; resetseed/zero outsidetimer. Renamed originalmain directlaunchaliases dormant. Julia pipeline! uses explicit first_column!/column_y!/last_entries!/columns! callbacks throughCUDA/KA/JACC with same twostream joins and finalcompletion; main timercalls fullpipeline. Source UInt32 fourlag additiveFibonacci recurrence modulo2^32, n16384,r16, fixedseedstate; no scalarFP/quality/MonteCarlo guarantee. Allocation/JACCsharedworkspaces/warmup/oracle/readback outside. Four sourcefamilies and source dependencyedges do not prove realizedstreamoverlap, throughput bottleneck or fullprofilerphase.',
'log2':'Native selected suite directly invokes compute_log for19approximationdepths each17repeats within timer, compiled with exact -DSTUDY_FIX_PRECISION=1 selecting study_corrected_kernel.h. Headerwrapper run_test candidates in original/corrected headers are not called by study and aliases disabled. Julia suite! invokes factorycallback log_cuda!/log_ka!/log_jacc! with explicit row,p for19depths and17repeats before synchronizedelapsed. Source Float32 repeatedsquare binarylogapproximation plus UInt64denominator shift (recorded correctedreference convention), NOT19hardwareprecision formats. Reset/inputecho/oracleaccuracy budget andguardchecks outside; alloldreferencequalifications preserved.',
'logprob':'Native selected launch invokes log_probs_kernel<float> followed by accumulate_log_probs within10repeat synchronizedtimer; originalincludedmain renamed/dormant. Julia studyfactory closure explicitly binds log_probs_kernel!/accumulate_kernel!, KA namedpair/JACC callbacks, and OwnedStudy times tenpaircalls afterreset. CUDAjl include_string only defines originaldeviceprefix before CPUroutine, so oldhostlaunchaliases uninvoked. Float32 stabilized logsoftmax of logits for providedtokenIDs plus sequence sum; source exp/log/reductions andInt32ids/lengths, batch64,maxlen32,vocab256. Setup/warmup/reset/readback/oracle outsidetimer; operator scope is scoreevaluation, not inference/token generation or language-modelquality.'}
a=s.index('why=');z=s.index('\nlock=',a);s=s[:a]+'why='+repr(why)+s[z:]
a=s.index("  if b=='kurtosis'and m=='cuda':");z=s.index('  ev=[];',a)
s=s[:a]+'''  overlays=[]
  assert not plan.get('capture_overlay_provenance',{}).get('patches'),('requires independent overlay review',b,m)
  if m=='cuda':
   paths.add(base/('study_cuda.cu'if b=='layernorm'else'study_native.cu'))
   if b=='log2':
    assert '-DSTUDY_FIX_PRECISION=1'in plan['build_command']
    paths.add(base/'study_corrected_kernel.h')
  else:
   if (base/'study_common.jl').is_file():paths.add(base/'study_common.jl')
   for arg in plan['profile_target_command']:
    if str(arg).endswith('.jl')and Path(arg).is_file():paths.add(Path(arg).resolve())
   if b in ['layout','logprob']:paths.add(W/'src/_portable_study/OwnedStudy.jl')
   if b=='lavaMD':
    paths.add(base/'study_types.jl')
    if m!='ka':paths.add(base/'study_body.jl')
   if b=='layernorm':paths.add(base/'layernorm_math.jl')
'''+s[z:]
a=s.index("   path=Path(s['source_callsite']['path']);legacy=");z=s.index("   updates['ROI_membership_proof']=",a)
s=s[:a]+'''   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and legacy and b!='layernorm')or(m=='cuda_jl'and b in ['layout','logprob']and path.name=='main.jl')or(m=='cuda'and b=='log2'and path.name=='study_corrected_kernel.h')
   unselected=False
   if b=='layout':
    assert plan['profile_target_command'][-2:]==['soa','10']
    unselected=(m=='cuda'and si==1)or(m in ['cuda_jl','ka']and si==0)
   kind='dormant_legacy_entrypoint_or_unused_header_wrapper'if dormant else'unselected_AoS_mode_branch'if unselected else'timed_semantic_family'
   included=not(dormant or unselected)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_uninvoked_legacy_wrapper_alias_or_captured_argv_unselected_AoS')
   if b=='layernorm'and m=='cuda'and si<2:
    template=4 if si==0 else 1
    updates.update(model_kernel_regex=r'\\blayernorm_forward_kernel0<\\(int\\)'+str(template)+r'>',regex_status='exact_source_Cmod4_UNROLL4_vs1_template_branch')
   if b=='layout'and m=='jacc':
    updates.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\\n]*\\bsoa_jacc_(?=[,<(])',resolved_callable_candidates=['soa_jacc!'],regex_status='exact_captured_soa_mode_source_ternary_dispatch')
'''+s[z:]
s=s.replace('Private batch8 source-only enrichment five named cases','Private batch9 source-only enrichment six named cases').replace('Private batch8 source-only enrichment','Private batch9 source-only enrichment')
(O/'prepare_proposals.py').write_text(s)
