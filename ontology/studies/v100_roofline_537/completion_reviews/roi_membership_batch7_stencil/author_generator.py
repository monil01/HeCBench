from pathlib import Path
O=Path(__file__).parent;s=Path('/tmp/roofline-completion-20261021/roi_membership_batch6_stencil/prepare_proposals.py').read_text();a=s.index('N=');z=s.index('\ndef sha',a);s=s[:a]+"N=set('gmm goulash gpp graphExecution gru'.split())"+s[z:]
why={
'gmm':'Selected gmm_run starts host Clock before cluster() and subtracts explicitly measured pre/poststage observation intervals. Included controller_adapted.inc routes all7source stage calls through gmm_dispatch. Native dispatch switch invokes named native kernels; Julia common C-ABI callback wraps devicefields and explicitly calls launch_gmm(stage,...), whose branch maps to actual CUDA/KA/JACC kernels. The configured_native/cluster.cu direct launch candidates are NOT included; selected host_driver includes controller_adapted.inc, not cluster.cu. Source Float32 Gaussian-mixture EM/model-order convergence/merge controller and seven distinct GPU stage operators; tested d2/n992/capacity6. Guard allocation, transfers and CPUcontrol remain part of net timed pipeline; explicit observation intervals excluded. Source inclusion is net-timer callgraph proof, not contiguous elapsed ROI or captured phase classification.',
'goulash':'Native selected study performs one gate warmup outside timer, then10updates from zero reset inside timer with sync before elapsed. Native included historical main renamed/uninvoked. Julia study(run!) invokes supplied explicit kernel!(gate,voltage,Int32(n)) callback in timed10loop; initial warmup/copy/reset and readback/oracle outside. Captured CUDAjl/JACC generic kernel signatures exactly match TWO Float64 arrays plusInt32; fullnames anchored, not host run! or bare dispatcher. KA saved signature has unexplained Float64 scalar in place of second devicearray and is separately withheld. Source polynomial/rational voltage-dependent gate target and exponential relaxation recurrence at4096fixedvoltages, not a whole cardiac model.',
'gpp':'Native selected launch invokes solver once per each4case inside synchronized timer after outputreset; renamed original main dormant. Julia gpp_study invokes supplied launch(c) after reset/sync and before syncelapsed; CUDAjl study explicit solver_kernel!, KA/JACC explicit reduce_gpp! delegates gpp_thread! with Float64 split-complex products/divisions/atomic reductions. OriginalCUDAjl main guarded false whenstudyincluded. This is scoped electronic-structure generalized plasmon-pole complex reduction, not fullGW or electronic-spectrum calculation. Outputreset/fill and input/output/guard/oracle readbacks outside timer; saved fillkernel samples stayunassigned, no full scientific sample-phase proof.',
'graphExecution':'Selected native segment copies pinned hostinput H2D,100times zeros partial and invokes reduce+reduceFinal, then D2H result. Exact segment is captured into graph before timing; each trial timer visits graphLaunch(executable) and direct segment separately with stream completion, sums both. Julia same captured segment closure invokes explicit launch_reductions callbacks and transfer/reset path; executable capture/instantiate outside timer, graphlaunch or CUDA.stream!(segment!,stream) inside. Source Float32 inputs accumulated into Float64 partial/final reductions; original renamed main candidates unused. Kernel family source membership applies to captured and ordinary segment paths, but first sampled invocation can be warmup/capture/other path; no attribution to timed graph-vs-stream segment or complete event census.',
'gru':'Native selected study launches gru_cell_forward<half,float,int> inside100repeat synchronized timer, originalmainrenamedunused. Julia study_main invokes prepare-returned closure via study_loop! inside same timer; CUDAjl study removes legacy exit(main(ARGS)), KA/JACC explicitly bind portable_gru!. Source Float16 input/state storage, Float32 gate/sigmoid/tanh/intermediate arithmetic, half output plus all5saved-state planes; widening to Float32 artifact occurs outside timer. Inputs represent precomputed gate-affine values, not included matrixmultiplications/fullrecurrenttraining. Setup/warmup/reference and readback excluded; no tensor-instruction inference from half storage.'}
a=s.index('why=');z=s.index('\nlock=',a);s=s[:a]+'why='+repr(why)+s[z:]
a=s.index('  overlays=[]');z=s.index('  ev=[];',a)
s=s[:a]+'''  if b=='goulash'and m=='ka':
   source=base/'goulash-julia-KA/main.jl'
   assert plan['execution_bindings'].get(str(source))==file(source)['sha256']
   withheld.append({'benchmark':b,'model':m,'reason':'Generic saved signature has one Float64 devicearray plusFloat64scalar+Int32; source kernel!(gate,voltage,n) takes two Float64devicearrays+Int32. No selected-instance lowering proof; no speculative fill/setup/math assignment. Numerical verdict unchanged.','actual_saved_identities':[x['identity']for x in obs],'selected_config':r['raw_config'],'selected_result':r['reference_result'],'selected_plan':file(pp),'selected_manifest':file(mp),'selected_derivation':file(op),'pinned_source_evidence':evidence(source)});continue
  overlays=[]
  assert not plan.get('capture_overlay_provenance',{}).get('patches'),('requires independent overlay review',b,m)
  if m=='cuda':
   paths.add(base/('host_driver.cu'if b=='gmm'else'study_native.cu'if b in ['goulash','graphExecution']else'study_cuda.cu'))
  elif (base/'study_common.jl').is_file():paths.add(base/'study_common.jl')
  if m!='cuda':
   for arg in plan['profile_target_command']:
    if str(arg).endswith('.jl')and Path(arg).is_file():paths.add(Path(arg).resolve())
  if b=='gmm':paths.update([base/'host_driver.cu',base/'controller_adapted.inc'])
  if b=='gpp'and m in ['ka','jacc']:paths.add(base/'reduction_math.jl')
'''+s[z:]
a=s.index("   path=Path(s['source_callsite']['path']);legacy=");z=s.index("   updates['ROI_membership_proof']=",a)
s=s[:a]+'''   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and (legacy or (b=='gmm'and path.name=='cluster.cu')))or(m=='cuda_jl'and b in ['gpp','gru']and path.name=='main.jl')
   warmup=(m=='cuda'and b=='goulash'and si==0)
   kind='dormant_legacy_entrypoint'if dormant else'warmup'if warmup else'timed_semantic_family'
   included=not(dormant or warmup)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_warmup_or_uninvoked_original_controller_or_legacy_main_alias')
   if b=='goulash'and m in ['cuda_jl','jacc']:
    assert len(obs)==1
    actual=obs[0]['identity']['Kernel Name']
    assert actual.count('CuDeviceArray<Float64')==2 and actual.endswith(', Int32)')
    updates.update(model_kernel_regex=r'^'+re.escape(actual)+r'$',resolved_callable_candidates=['kernel!'],regex_status='exact_full_typed_saved_name_twoFloat64arrays_Int32_source_explicit_callback_not_bare_dispatcher')
'''+s[z:]
s=s.replace('Private batch6 source-only enrichment six named cases','Private batch7 source-only enrichment five named cases').replace('Private batch6 source-only enrichment','Private batch7 source-only enrichment')
(O/'prepare_proposals.py').write_text(s)
