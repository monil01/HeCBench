from pathlib import Path
import json
O=Path(__file__).parent
p=Path('/tmp/roofline-completion-20261021/roi_membership_batch4_stencil/prepare_proposals.py')
s=p.read_text();a=s.index('N={');z=s.index('\ndef sha',a);s=s[:a]+"N=set('divergence doh dpid dslash dwconv dwconv1d dxtc2 eigenvalue expdist extrema'.split())"+s[z:]
why={
'divergence':'Selected native timed repeat calls divergence_sphere_gpu<4,double>, whose divergence.hpp body performs CPU contravariant transform, device allocation/H2D, div_kernel, D2H/free. Julia full_operation performs analogous host/device work and calls actual study_prepare callback inside timed repeated full_operation. KA binds DIV_KERNEL and JACC divergence_kernel! explicitly. Both NP4 manufactured polynomial cases are timed; initial warmup and subsequent oracle outside. This is Float64 divergence of a transformed vector field with original Earth-radius factor, not pure kernel elapsed or full atmosphere solver.',
'doh':'Selected native launch lambda directly calls hessian_matrix_det for four image cases and four sigma values within synchronized100-repeat timers after guarded reset. Julia doh_study invokes passed launch! in same timed loop; CUDAjl study binds hessian_matrix_det_kernel!, KA explicitly instantiates DOH_KA and JACC explicitly binds doh_kernel!. CUDAjl included legacy main guarded false for study.jl; native renamed main dormant. Float32 box-Hessian determinant operator; initialization, warmup, input echo and oracle outside.',
'dpid':'Native selected launch directly invokes kernelGuidance then kernelDownsampling, repeated10times inside timer for two patch regimes. kernels.cu launch candidates belong unused downsample wrappers, not this direct study call. Julia factory returns closure with explicit guidance and downsampling callbacks, shared dpid_study invokes it after reset/warmup and before sync/elapsed. CUDAjl include_string removes final legacy main() before study factory. RGB tuple/UInt8 output with floating guidance weights, source-selected lambda bound by captured environment; full image resize operator scope only.',
'dslash':'Native selected study calls back(c) and synchronizes BEFORE timer, then100hop(c) calls inside timer. Julia common similarly launches back_kernel! before time_ns then100synchronized launch_dslash calls; actual model wrappers explicitly bind hopping_kernel!. Backwardlink setup is excluded even though profiles can sample it; original kernels.cu wrapper launches are unused by selected direct lambdas. Source Float32 staggered lattice hopping with SU3 links, not a full QCD solver; oracle/readback/reset outside.',
'dwconv':'Native Case.launch source selects template3,1,0 according to k3,1,other; all12 multiplier/kernel cases timed for10repeats after reset/warmup. Included original main renamed dormant. Julia dwconv_study invokes factory-returned closure in timed10loop; CUDAjl study strips final exit(main()), portable callbacks explicitly bind portable_dwconv! -> dwconv_lane!. Float32 depthwise convolution plus bias; input/weights echo and oracle after timing, not training pipeline. Exact native template selectors distinguish three source branches, no duplicate family aliases.',
'dwconv1d':'Native selected run calls cuda_forward then cuda_backward; timex.cu wrappers launch kernel_forward/backward and synchronize before elapsed. Julia run(c) invokes explicit launch callback with forward!/backward! within same synchronized timed path for three fixture cases; KA handles and JACC callbacks bind names explicitly. JACC shared templates allocated outside timer, launch specs allocate source shared buffers. Float32 causal time convolution and gradients, not full model training. Fixture allocation/warmup and full output/guard checks outside.',
'dxtc2':'Native compress warmup candidate precedes all timers; timed second compress call repeats10times for full12blocks and5blocks atoffset2. Renamed original main dormant. Julia study_main invokes explicit launch_compress callback within same10-repeat timer; actual main binds compress_study! via CUDA/KA/JACC dispatch. Guardreset/inputecho/MSE and packed-stream validation outside. Source mixed floating endpoint search and integer DXT/BC bit packing; no scalarFP work implied by compressed-output dtype.',
'eigenvalue':'Native study repeated launch closure reinitializes two interval buffers, copies H2D, then loops explicit count/recalculate kernels with host completion check and D2H until tolerance; all of closure inside timed10solve repeats. Included original main dormant, separate original runKernels diagnostic occurs BEFORE measured series. Julia factory returns explicit two-kernel pair!, eigenvalue_study builds restarted closure and OwnedStudy.run_trials! times it; CUDAjl study only includes original prefix before CPU-helper function. Float32 Sturm bisection spectral enclosure; host convergence/transfers remain inside ROI, no pure GPU-kernel elapsed claim.',
'expdist':'Native launch closure invokes distance<T> followed by reduce_cross_term<T> for all18 Float32/Float64 shape/fixture paths within10-repeat timers. Julia exp_study invokes factory closure; CUDAjl separately binds k=@cuda launch=false distance_jl! and r=...reduce_jl!, then calls both; KA named handles and JACC callbacks explicit. Allocation, initial buffer reads, warmup, guards and independent oracle outside. Pairwise exponential-weighted distance partials and global reduction, source SFU/atomic use not counter-derived bottleneck.',
'extrema':'Native Data.launch branches by source dim to relextrema_1D<T>/relextrema_2D<T> and source path visits clip/wrap/order/axis choices. Julia launch_extrema same explicit branch and callback bindings;100repeated calls inside each synchronized path timer. Guards/reset before, complete masks/inputecho/oracle after. Four source storage types Int32/Int64/Float32/Float64 use comparisons/Boolean masks, not ordinary scalar add/mul/FMA. No physiological peak interpretation or general signal quality claim.'}
a=s.index('why={');z=s.index('\nlock=',a);s=s[:a]+'why='+repr(why)+s[z:]
a=s.index("  if m=='cuda':paths.add");z=s.index('  ev=[];',a)
s=s[:a]+'''  if m=='cuda':paths.add(base/('study_cuda.cu'if b in ['divergence','doh','dslash','dxtc2','extrema']else'study_native.cu'))
  else:
   paths.add(base/'study_common.jl')
   if b=='eigenvalue':paths.add(W/'src/_portable_study/OwnedStudy.jl')
   for name in {'doh':['hessian_math.jl'],'dpid':['study_device.jl'],'dslash':['study_device.jl'],'dwconv':['study_device.jl']}.get(b,[]):
    if (base/name).is_file():paths.add(base/name)
'''+s[z:]
a=s.index("   path=Path(s['source_callsite']['path']);legacy=");z=s.index("   updates['ROI_membership_proof']=",a)
s=s[:a]+'''   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and legacy and b not in ['divergence','dwconv1d'])or(m=='cuda_jl'and b in ['doh','dpid','dwconv','eigenvalue']and path.name=='main.jl')
   warmup=(m=='cuda'and b=='dxtc2'and si==0)
   setup=(b=='dslash'and((m=='cuda'and si==0)or(m!='cuda'and si==0)))
   kind='dormant_legacy_entrypoint'if dormant else'warmup'if warmup else'backward_link_setup_before_timer'if setup else'timed_semantic_family'
   included=not(dormant or warmup or setup)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_same_family_warmup_or_dormant_alias_or_explicit_pre_timer_setup')
   if b=='dwconv'and m=='cuda'and included:
    template=[3,1,0][si]
    updates.update(model_kernel_regex=r'\\bconv_depthwise2d_forward_kernel<\\(int\\)'+str(template)+r',',regex_status='exact_source_template_branch_selector_no_broad_alias',resolved_callable_candidates=['conv_depthwise2d_forward_kernel<'+str(template)+',float,float,int>'])
   if b=='expdist'and m=='cuda_jl':
    callback=['distance_jl!','reduce_jl!'][si]
    updates.update(model_kernel_regex=r'\\b'+callback.replace('!','_')+r'(?=[(<])',regex_status='separate_source_k_and_r_launch_false_handles_exact_callback',resolved_callable_candidates=[callback])
   if b=='doh'and m=='cuda_jl'and included:
    updates.update(model_kernel_regex=r'\\bhessian_matrix_det_kernel_(?=[(<])',resolved_callable_candidates=['hessian_matrix_det_kernel!'],regex_status='explicit_cuda_study_callback_not_host_study_wrapper')
'''+s[z:]
s=s.replace("  parents[str(parent)]={'compressed_sha256'",'''  if not any(x['matched_stage_ids']for x in matches):
   withheld.append({'benchmark':b,'model':m,'reason':'No positive saved included-family match; proposed model withheld','selected_observations':matches});continue
  parents[str(parent)]={'compressed_sha256' ''').replace("'compressed_sha256' :","'compressed_sha256':")
s=s.replace('Private batch4 source-only enrichment','Private batch5 source-only enrichment')
(O/'prepare_proposals.py').write_text(s)
