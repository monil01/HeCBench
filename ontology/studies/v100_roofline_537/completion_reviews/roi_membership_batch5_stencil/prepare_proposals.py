import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set('divergence doh dpid dslash dwconv dwconv1d dxtc2 eigenvalue expdist extrema'.split())
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
why={'divergence': 'Selected native timed repeat calls divergence_sphere_gpu<4,double>, whose divergence.hpp body performs CPU contravariant transform, device allocation/H2D, div_kernel, D2H/free. Julia full_operation performs analogous host/device work and calls actual study_prepare callback inside timed repeated full_operation. KA binds DIV_KERNEL and JACC divergence_kernel! explicitly. Both NP4 manufactured polynomial cases are timed; initial warmup and subsequent oracle outside. This is Float64 divergence of a transformed vector field with original Earth-radius factor, not pure kernel elapsed or full atmosphere solver.', 'doh': 'Selected native launch lambda directly calls hessian_matrix_det for four image cases and four sigma values within synchronized100-repeat timers after guarded reset. Julia doh_study invokes passed launch! in same timed loop; CUDAjl study binds hessian_matrix_det_kernel!, KA explicitly instantiates DOH_KA and JACC explicitly binds doh_kernel!. CUDAjl included legacy main guarded false for study.jl; native renamed main dormant. Float32 box-Hessian determinant operator; initialization, warmup, input echo and oracle outside.', 'dpid': 'Native selected launch directly invokes kernelGuidance then kernelDownsampling, repeated10times inside timer for two patch regimes. kernels.cu launch candidates belong unused downsample wrappers, not this direct study call. Julia factory returns closure with explicit guidance and downsampling callbacks, shared dpid_study invokes it after reset/warmup and before sync/elapsed. CUDAjl include_string removes final legacy main() before study factory. RGB tuple/UInt8 output with floating guidance weights, source-selected lambda bound by captured environment; full image resize operator scope only.', 'dslash': 'Native selected study calls back(c) and synchronizes BEFORE timer, then100hop(c) calls inside timer. Julia common similarly launches back_kernel! before time_ns then100synchronized launch_dslash calls; actual model wrappers explicitly bind hopping_kernel!. Backwardlink setup is excluded even though profiles can sample it; original kernels.cu wrapper launches are unused by selected direct lambdas. Source Float32 staggered lattice hopping with SU3 links, not a full QCD solver; oracle/readback/reset outside.', 'dwconv': 'Native Case.launch source selects template3,1,0 according to k3,1,other; all12 multiplier/kernel cases timed for10repeats after reset/warmup. Included original main renamed dormant. Julia dwconv_study invokes factory-returned closure in timed10loop; CUDAjl study strips final exit(main()), portable callbacks explicitly bind portable_dwconv! -> dwconv_lane!. Float32 depthwise convolution plus bias; input/weights echo and oracle after timing, not training pipeline. Exact native template selectors distinguish three source branches, no duplicate family aliases.', 'dwconv1d': 'Native selected run calls cuda_forward then cuda_backward; timex.cu wrappers launch kernel_forward/backward and synchronize before elapsed. Julia run(c) invokes explicit launch callback with forward!/backward! within same synchronized timed path for three fixture cases; KA handles and JACC callbacks bind names explicitly. JACC shared templates allocated outside timer, launch specs allocate source shared buffers. Float32 causal time convolution and gradients, not full model training. Fixture allocation/warmup and full output/guard checks outside.', 'dxtc2': 'Native compress warmup candidate precedes all timers; timed second compress call repeats10times for full12blocks and5blocks atoffset2. Renamed original main dormant. Julia study_main invokes explicit launch_compress callback within same10-repeat timer; actual main binds compress_study! via CUDA/KA/JACC dispatch. Guardreset/inputecho/MSE and packed-stream validation outside. Source mixed floating endpoint search and integer DXT/BC bit packing; no scalarFP work implied by compressed-output dtype.', 'eigenvalue': 'Native study repeated launch closure reinitializes two interval buffers, copies H2D, then loops explicit count/recalculate kernels with host completion check and D2H until tolerance; all of closure inside timed10solve repeats. Included original main dormant, separate original runKernels diagnostic occurs BEFORE measured series. Julia factory returns explicit two-kernel pair!, eigenvalue_study builds restarted closure and OwnedStudy.run_trials! times it; CUDAjl study only includes original prefix before CPU-helper function. Float32 Sturm bisection spectral enclosure; host convergence/transfers remain inside ROI, no pure GPU-kernel elapsed claim.', 'expdist': 'Native launch closure invokes distance<T> followed by reduce_cross_term<T> for all18 Float32/Float64 shape/fixture paths within10-repeat timers. Julia exp_study invokes factory closure; CUDAjl separately binds k=@cuda launch=false distance_jl! and r=...reduce_jl!, then calls both; KA named handles and JACC callbacks explicit. Allocation, initial buffer reads, warmup, guards and independent oracle outside. Pairwise exponential-weighted distance partials and global reduction, source SFU/atomic use not counter-derived bottleneck.', 'extrema': 'Native Data.launch branches by source dim to relextrema_1D<T>/relextrema_2D<T> and source path visits clip/wrap/order/axis choices. Julia launch_extrema same explicit branch and callback bindings;100repeated calls inside each synchronized path timer. Guards/reset before, complete masks/inputecho/oracle after. Four source storage types Int32/Int64/Float32/Float64 use comparisons/Boolean masks, not ordinary scalar add/mul/FMA. No physiological peak interpretation or general signal quality claim.'}
lock=Path('/tmp/hecbench-roofline-metadata.lock').open('r');fcntl.flock(lock,fcntl.LOCK_SH)
proposals=[];sources={};parents={};selected=[];withheld=[]
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
  base=W/'src'/f'{b}-julia'; paths={Path(s['source_callsite']['path']).resolve() for s in r['stages']}
  overlays=[]
  for i,patch in enumerate(plan.get('capture_overlay_provenance',{}).get('patches',[])):
   if 'text'not in patch:continue
   text=patch['text'];digest=sha(text.encode());assert digest==patch['copy_sha256']==plan['execution_bindings'][patch['copy']]
   original=Path(patch['original']);original_bytes=original.read_bytes();assert sha(original_bytes)==patch['original_sha256']==plan['execution_bindings'][patch['original']]
   transformed=original_bytes.decode();transform_steps=[]
   for change in patch['changes']:
    before,after=change['before'],change['after'];assert transformed.count(before)==1
    assert any(token in before for token in ['rawdir=joinpath(', 'read(joinpath(@__DIR__,"fixture.bin"))', 'include(joinpath(@__DIR__,"..","study_common.jl"))'])
    transform_steps.append({'before':before,'after':after,'matched_occurrences':1,'scope':'output directory, same canonical fixture read root, or include of exact hash-bound overlay common only'})
    transformed=transformed.replace(before,after,1)
   assert transformed==text
   inverse=text
   for change in reversed(patch['changes']):assert inverse.count(change['after'])==1;inverse=inverse.replace(change['after'],change['before'],1)
   assert inverse.encode()==original_bytes
   overlays.append({'actual_executed_source_path':patch['copy'],'actual_executed_sha256':digest,'retained_in_selected_plan':True,'retained_text':text,'original_bound_path':patch['original'],'original_sha256':patch['original_sha256'],'original_overlay_file_currently_exists':Path(patch['copy']).is_file(),'selected_plan':file(pp),'text_JSON_pointer':'/capture_overlay_provenance/patches/'+str(i)+'/text','qualification':'Exact complete executed-copy text retained in immutable selectedplan; no scratch recreation or current-source substitution.','exact_original_to_copy_transform':transform_steps,'forward_transform_byte_equal':True,'inverse_transform_recovers_original_byte_equal':True,'algorithm_timer_callback_and_arithmetic_unchanged_beyond_path_roots':True})
  if m=='cuda':paths.add(base/('study_cuda.cu'if b in ['divergence','doh','dslash','dxtc2','extrema']else'study_native.cu'))
  else:
   paths.add(base/'study_common.jl')
   if b=='eigenvalue':paths.add(W/'src/_portable_study/OwnedStudy.jl')
   for name in {'doh':['hessian_math.jl'],'dpid':['study_device.jl'],'dslash':['study_device.jl'],'dwconv':['study_device.jl']}.get(b,[]):
    if (base/name).is_file() and not(m=='cuda_jl'and b in ['doh','dpid','dwconv']):paths.add(base/name)
  ev=[];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for p in sorted(paths):
   e=evidence(p)
   assert bindings.get(str(p.resolve()))==e['sha256'],('selected source binding gap',b,m,str(p))
   ev.append(e);sources[e['path']]=file(p)
  patches=[]
  for si,s in enumerate(r['stages']):
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   dormant=(m=='cuda'and legacy and b not in ['divergence','dwconv1d'])or(m=='cuda_jl'and b in ['doh','dpid','dwconv','eigenvalue']and path.name=='main.jl')
   warmup=(m=='cuda'and b=='dxtc2'and si==0)
   setup=(b=='dslash'and((m=='cuda'and si==0)or(m!='cuda'and si==0)))
   kind='dormant_legacy_entrypoint'if dormant else'warmup'if warmup else'backward_link_setup_before_timer'if setup else'timed_semantic_family'
   included=not(dormant or warmup or setup)
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_same_family_warmup_or_dormant_alias_or_explicit_pre_timer_setup')
   if b=='dwconv'and m=='cuda'and included:
    template=[3,1,0][si]
    updates.update(model_kernel_regex=r'\bconv_depthwise2d_forward_kernel<\(int\)'+str(template)+r',',regex_status='exact_source_template_branch_selector_no_broad_alias',resolved_callable_candidates=['conv_depthwise2d_forward_kernel<'+str(template)+',float,float,int>'])
   if b=='expdist'and m=='cuda_jl':
    callback=['distance_jl!','reduce_jl!'][si]
    updates.update(model_kernel_regex=r'\b'+callback.replace('!','_')+r'(?=[(<])',regex_status='separate_source_k_and_r_launch_false_handles_exact_callback',resolved_callable_candidates=[callback])
   if b=='doh'and m=='cuda_jl'and included:
    updates.update(model_kernel_regex=r'\bhessian_matrix_det_kernel_(?=[(<])',resolved_callable_candidates=['hessian_matrix_det_kernel!'],regex_status='explicit_cuda_study_callback_not_host_study_wrapper')
   updates['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':[{'path':e['path'],'sha256':e['sha256'],'timer_lines':[{'line':e['start_line']+i,'text':line}for i,line in enumerate(e['snippet'].splitlines())if any(v in line for v in ['time_ns()', 'steady_clock::now()', 'run_trials!(', 'bm_study(factory)', 'exit(study_main(', 'bezier_study(', 'box_study(factory)'])]}for e in ev],'captured_overlay_source_evidence':overlays,'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'selected_environment_overrides':plan['environment_overrides'],'precision_scope':'Source types/operations/Val branches are in exact evidence; selectedenvironment retained. Float32/Float64/mixed integer/conversion/SFU/clock semantics are distinct from actual counter instruction class; no arithmetic mode inferred from benchmark name.','limitations':['Source inclusion only: first captured invocation may be warmup/setup; sample phase remains unproved.','No complete elapsed-ROI or complete launch census, multiplicity extrapolation, numerical recertification or causal bottleneck claim.','Alias nulling avoids repeated same-family candidates; it does not remove source calls or hide conditional branches.']}
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
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'schema_version':1,'scope':'Private batch5 source-only enrichment ten named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
