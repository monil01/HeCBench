import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set(['moe-align', 'moe-sum', 'moe', 'morphology', 'mpc', 'mr', 'mrc', 'mriQ', 'mt', 'mtf', 'multimaterial', 'multinomial', 'muon', 'murmurhash3', 'myocyte', 'opticalFlow'])
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
why={'moe-align': 'Selected source dispatch visits five token/expert shapes, using smallbatchkernel or generalalign plus scatter pipeline; Julia callbacks explicitly implement those branches inside three-repeat perfixture timer summed. Int32token/expert IDs, prefixcounts, padded slots and routingindices, not ordinaryFP. Reset/sourceoracle/readbacks outside. Algorithmically necessary paddedwork and expertmapping constraints preserved; not expertMLP compute or trainedMoEquality.', 'moe-sum': 'Captured argv vec4 selects only vec4callback/native templateK3, summing three supplied expert rows for512tokensx1024hidden. Source Float32vector4payload/accumulation, not tensor orFP16 inference. Julia factory chooses same vecflag, OwnedStudy repeats10times afterreset with completion. Scalarbranch and standaloneoldmain paths excluded; exactJACCcallbackresolved from factory. Partialexpertaggregation only, not routing/training/weightedgatingmodel.', 'moe': 'Native and Julia closure launch softmax plus topKselection over supplied routerlogits within percase20repeat means. Source Float32 exp/logitnormalization/reductions and Int32expertIDs, fixed ties/selectedrow qualification from existing contract retained. Reset/warmup/inputcapture/fulloracle after/before timer. This is gating operator scope, not fullMoE inference/training or quality.', 'morphology': 'Source cycle timed10repeats across two510x510fixtures: horizontal+vertical dilation then erosion with two temporaryzeroresets insidecycle. Native vHGW5tap source kernels and repairedJulia naive separableaxis kernel have same recorded boundarysemantics but differentalgorithmicmapping. UInt8max/min and integercoordinates, not ordinaryFP. Native enumERODE0,DILATE1 explicit template selectors; Julia runtimeop/axis samefamily aliases retainedwithoutduplicateattribution. Warmup/inputrestore/oracleoutside; source-definedclosing, not arbitraryimagequality claims.', 'mpc': 'Selected partial_corrected caseset executes original-style persistent160block1024thread ring compression and decompression with offsetreset and completion in componenttimers. UInt32bitstream/predictor and Float32inputbitpatterns; no FPwork inferred fromstorage. Hostcapacity/partialtail qualifiedreference boundary remains. Source exactvariablelengthstream/fullroundtrip and guardchecks outside. JACC generatedwrapper launchbound path lacks an existing sourcecallsitecandidate, held rather than newstage or unsupportedclaim.', 'mr': 'Captured odd,efficient selects mr32_eff countkernel only; mr32_sf unselected. Evidence kernel computing fullindividualcodes is untimed aftercountwindow and warmup. Julia countcallback runsinside perwidth10launch/sync sumtimer; evidencecallbackoutside. UInt32 modularinteger threebase MillerRabin-style primalityoperator and Int32count, notFP or a general primality/securitycertificate. Fullcodes/countoracle outside and no extrawork attribution.', 'mrc': 'Three supplied margin-ranking-gradient algorithms separately timed100repeats with completion, reset outside. ExplicitJulia mode dispatcher and native switch choose same branchwiseFloat32linearhinge gradients with Int32labels, including exacthingeequality. MRC here is marginrankingloss, not magneticresonance imaging. Warmup/oracle/readbackoutside; not fullautodiff/trainingmodel.', 'mriQ': 'Selected native directPhiMag kernel and tiledQ kernel, Julia explicitlaunchphi/launchq callbacks, belong fivecase sum11operator windows. Q windows include host-to-device tile upload and completion; nativeconstantck vsJuliaordinarydevicearray mapping disclosed. SourceFloat32 complexweighted nonuniformFourier phase accumulation with trig/SFU, Float64 independentoperatororacle; not reconstructedclinicalMRIimage or inversepipeline. Outputreset/phi-readback/oracles outside. OriginalcomputeQwrapper hostcalls unused.', 'mt': 'Primarysourcefull4096RNGstreams x5860outputpair MT+BoxMuller averaged3repeats; twoadditional shorterseedcases tested separately, not pooledprimarytime. Native loop mt+boxmuller and authoredJulia callbacks inside sourcepairtimers; posttimeduniformMT replay explicitlyexcluded. UInt32RNGrecurrence plusFloat32Gaussiantransform/log/sqrt/trig, mixedinteger/SFUwork not oneFPceiling. Parameters/seedreset/fulluniform/Gaussian/guards outsideprimarytimer. Finitegeneratorchecks not distributionquality guarantee.', 'mtf': 'Source-specific corrected full classicalmove-to-front secondary: Julia hostalphabetinitialize/update plusperbyteH2D alphabetcopy, GPUfind_symbol search andrankD2H inside completeencode timer. Explicitsearch callback atselectedexistingstage; native usesoperation/libraryplaceholder and remainsunresolved. UInt8alphabet/wordandInt32rank, not ordinaryFP or pureruntimekernelthroughput. Fullindependentlastpositionoracle/decoder/guards outside. Originalnative allocation/semanticdefects remainhistorical, no untouchedtransformclaim.', 'multimaterial': 'Selected mode2 compacttensor fixture times threealgorithms: pure/mixed density, pure/mixed pressure andneighborcell averaging withcompletionboundaries; fivenative study-lambda kernels/Julia launch_phase! explicitcallbacks. Float64fraction/fieldarithmetic andInt32compactlinks. Correctedsource/interfacemetadataqualifications preserved; standalonecorrected/originalcompact hostdriveraliases unused. Input/load/reset/oracleoutside. Physicalmultimaterial operators, not fullhydrodynamics timeintegration or arbitraryallocationlayout comparison.', 'multinomial': 'Selected sampleMultinomialOnce and explicitJulia sample_multinomial_once_kernel! callback inside OwnedStudy repeatedlaunch/completion timer. Float32CDF/reductioncomparisons andInt32selectedcategory for provided samples/distributions; deterministicmargin/tolerancecontract retained. Setup/inputgeneration andsourceCPUoracleoutside, no probabilisticfrequencycalibration or freshRNGqualityclaim. Oldmainwarmup/benchmarkaliases uncalled.', 'muon': 'Fullsource fourshape optimizer fixture runs2steps percase withmomentum,normalization,fiveNewtonSchulziterations andparameterupdate inside synchronizedtimer. Explicitcallbackfamilies plus pinnedvendorGEMM; source Float32momentum/parameters/reduction withBF16storediterates inUInt16 andFP32-contribution convertedBF16. Vendorinternal arithmetic/SMfamily not inferredfromAPIorBF16storage, rawvendorrowsremainunassigned. Reset/oracle/readbackoutside. Sourceexactreshape/talltransposebranches allvisited acrossfourcases; no trainedmodelconvergenceclaim. JACCgenericlaunch mappedonlyexplicit8callbacklist; doesnotmapvendorGEMM.', 'murmurhash3': 'Selected study calls MurmurHash3_x64_128_kernel and exactJuliacallback overfourvariablelength fixturepaths in10repeat windows withcompletion. IntegerUInt64multiply/xor/rotateandUInt32length/seed, UInt8bytes; no scalarFP inference. Warmup/reset andindependentCPUhash/fullguard/outputoracle outside. Boundedhashoperator, nosecurityproperty or collisionqualitycertificate. Originalwrapperhostlaunch unused.', 'myocyte': 'Selected timedR20 kernel(int,float*,float*,float*,float*) launch2x32 computesoriginalECC/CaM componentRHS forfourfrozenstate/timeprobes; nativefp=float andnarrowcapturedsignature explicit. Firstnativewarmup andoldmaster launchunused. Julia factorysame2blockkernelbody mapping. Float32complextranscendental/nonlinearstateoperator plusintegerbranches; correctedCaMreference preserved. study_couple32 hostcoupling andindependentprobeoracle occur AFTERtimer, so not timedcompletecoupledODEintegrator or clinicalmodelvalidation. Inputsreset/readbacks/guardsoutside.', 'opticalFlow': 'Selected native ComputeFlowCUDA and Julia flow! primarycase0 run640x480 fivelevels/3warps/500Jacobi fullpipeline timer includingoriginalalloc/copies/free andhostloop; input/outputobservations pause timer AFTERdevicecompletion. Sixsourcefamilies downscale/upscale/warp/derivatives/Jacobi/add explicitlycalledthroughwrappers/dispatch. Float32flow/interpolation withintegerpitch/coordinates; boundarypolicies source-vslegacyCPU retained. Identical-image/primitivecontrols afterprimarytimer reusefamilies, so firstsamplemaybewarmup/control ratherthanprimary; no timedphaseproof. Scopedopticalflowoperator, no physicalscene-groundtruthaccuracy claim.'}
lock=Path('/tmp/hecbench-roofline-metadata.lock').open('r');fcntl.flock(lock,fcntl.LOCK_SH)
proposals=[];sources={};parents={};selected=[];withheld=[]
legacy_path=S/'ROI_REVIEW.json.gz';legacy_packed=legacy_path.read_bytes();legacy_raw=gzip.decompress(legacy_packed);legacy_doc=json.loads(legacy_raw);legacy_receipt_path=S/'ROI_REVIEW_RECEIPT.json';legacy_receipt=json.loads(legacy_receipt_path.read_bytes())
assert sha(legacy_packed)==legacy_receipt['sha256'] and sha(legacy_raw)==legacy_receipt['uncompressed_sha256']
assert not any(r['benchmark']in N for r in legacy_doc['records']), 'Legacy authority overlaps proposed names'
legacy_authority={'path':str(legacy_path),'compressed_sha256':sha(legacy_packed),'uncompressed_sha256':sha(legacy_raw),'receipt':file(legacy_receipt_path),'none_of_selected_names_present':True}

index_path=S/'roi_reviews/INDEX.json';index_bytes=index_path.read_bytes();index_doc=json.loads(index_bytes)
index_rows={str((W/x['path']).resolve()):x for x in index_doc['cohorts']}

for parent in sorted((S/'roi_reviews').glob('*.json.gz')):
 pb=parent.read_bytes();raw=gzip.decompress(pb);doc=json.loads(raw)
 assert index_rows[str(parent)]['compressed_sha256']==sha(pb)and index_rows[str(parent)]['raw_sha256']==sha(raw)
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
  overlays=[]
  if plan.get('capture_overlay_provenance',{}).get('patches'):
   withheld.append({'benchmark':b,'model':m,'reason':'Executed overlay requires separate inverse-source proof; not promoted'});continue
  # Quote only actual prospectively pinned source bytes, including helper/controller paths.
  paths=set()
  for k in plan['execution_bindings']:
   p=Path(k)
   if p.suffix in ('.jl','.cu','.cuh','.h','.hpp','.cpp','.c','.inc') and (str(base)+'/' in str(p) or '/src/'+b+'-cuda/' in str(p) or str(p).endswith('/src/_portable_study/OwnedStudy.jl')):
    paths.add(p.resolve())
  if not paths:
   withheld.append({'benchmark':b,'model':m,'reason':'No pinned actual source text available'});continue
  ev=[];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for p in sorted(paths):
   e=evidence(p)
   assert bindings.get(str(p.resolve()))==e['sha256'],('selected source binding gap',b,m,str(p))
   ev.append(e);sources[e['path']]=file(p)
  patches=[];canonical_families={}
  for si,s in enumerate(r['stages']):
   if not s.get('source_callsite'):continue
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   excluded=None
   if m=='cuda':
    if b=='moe-align'and si>=3:excluded='uninvoked_original_hostdriver'
    elif b=='moe-sum':excluded='argv_unselected_scalar_branch'if si==1 else'uninvoked_original_hostdriver'if si>=2 else None
    elif b=='morphology'and si>=4:excluded='uninvoked_original_hostwrapper'
    elif b=='mpc'and si>=2:excluded='uninvoked_original_hostdriver'
    elif b=='mr':excluded='argv_unselected_sf_branch'if si==1 else'untimed_full_evidence_capture'if si==2 else'uninvoked_original_hostdriver'if si>=3 else None
    elif b=='mrc'and si>=3:excluded='uninvoked_original_hostdriver'
    elif b=='mriQ'and si>=2:excluded='uninvoked_original_hostwrapper'
    elif b=='mt':excluded='untimed_uniform_replay'if si==2 else'uninvoked_original_hostwrapper'if si>=3 else None
    elif b=='multimaterial'and si>=5:excluded='uninvoked_original_or_corrected_standalone_hostdriver'
    elif b=='multinomial'and si>=1:excluded='uninvoked_original_hostdriver'
    elif b=='muon'and si>=8:excluded='uninvoked_original_hostdriver'
    elif b=='murmurhash3'and si==1:excluded='uninvoked_original_hostwrapper'
    elif b=='myocyte':excluded='untimed_warmup'if si==0 else'uninvoked_original_master'if si>=2 else None
   else:
    if b=='moe-sum':excluded='argv_unselected_scalar_branch'if(m=='cuda_jl'and si==1)or(m=='ka'and si==0)else'uninvoked_original_hostwrapper'if m=='cuda_jl'and si>=2 else None
    if b=='morphology'and m=='cuda_jl'and si>=2:excluded='uninvoked_original_hostdriver'
    if b=='mr'and si==1:excluded='untimed_full_evidence_capture'
    if b=='mrc'and m=='cuda_jl'and si>=3:excluded='uninvoked_original_hostdriver'
   kind=excluded or 'timed_semantic_family';included=excluded is None
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_untimed_audit_exclusion_proved'if excluded in ['untimed_full_evidence_capture','untimed_uniform_replay']else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_'+excluded)
   if included and b=='morphology'and m=='cuda':
    axis='horiz'if si%2==0 else'vert';morph_op=1 if si<2 else 0
    updates.update(model_kernel_regex=r'\bvhgw_'+axis+r'<\(MorphOpType\)'+str(morph_op)+r'>',regex_status='exact_source_enum_DILATE1_ERODE0_specialization')
   if included and b=='moe-sum'and m=='jacc':
    assert plan['profile_target_command'][-1]=='vec4'
    updates.update(model_kernel_regex=r'_parallel_for_cuda[^\n]*\bmoe_vec4_jacc_(?=[,<(])',resolved_callable_candidates=['moe_vec4_jacc!'],regex_status='exact_argv_vec4_factory_callback')
   if included and b=='multinomial'and m=='cuda_jl':updates.update(model_kernel_regex=r'\bsample_multinomial_once_kernel_(?=[(<])',resolved_callable_candidates=['sample_multinomial_once_kernel!'],regex_status='exact_selectedstudy_kernel_not_hostlaunchwrapper')
   if included and b=='muon'and m=='cuda_jl'and si in [2,3,4,5]:
    name={2:'sumsq_cuda!',3:'normalize_cuda!',4:'combine_cuda!',5:'update_cuda!'}[si]
    updates.update(model_kernel_regex=r'\b'+name.replace('!','_')+r'(?=[(<])',resolved_callable_candidates=[name],regex_status='exact_source_ordered_optimizer_callback')
   if included and b=='muon'and m=='jacc':updates.update(resolved_callable_candidates=['momentum_transpose_jacc!','momentum_jacc!','sumsq_jacc!','normalize_jacc!','combine_jacc!','update_jacc!','transpose_parameter_jacc!','parameter_jacc!'],regex_status='exact_run_function_eight_authored_callbacks_no_vendorGEMM_identity')
   if included and b=='myocyte'and m=='cuda':updates.update(model_kernel_regex=r'^kernel\(int, float \*, float \*, float \*, float \*\)$',regex_status='exact_source_fpfloat_kernel_signature_and_timed_driver_launch')
   regex=updates.get('model_kernel_regex',s.get('model_kernel_regex'))
   if included and regex:
    if regex in canonical_families:updates.update(model_kernel_regex=None,alias_of_stage_id=canonical_families[regex],regex_status='same_named_source_family_callsite_alias_no_duplicate_counterattribution')
    else:canonical_families[regex]=s['stage_id']
   updates['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':[{'path':e['path'],'sha256':e['sha256'],'timer_lines':[{'line':e['start_line']+i,'text':line}for i,line in enumerate(e['snippet'].splitlines())if any(v in line for v in ['time_ns()', 'steady_clock::now()', 'Clock::now()', 'excluded+=', 'run_trials!(', 'bm_study(factory)', 'exit(study_main(', 'bezier_study(', 'box_study(factory)'])]}for e in ev],'captured_overlay_source_evidence':overlays,'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'selected_environment_overrides':plan['environment_overrides'],'precision_scope':'Source types/operations/Val branches are in exact evidence; selectedenvironment retained. Float32/Float64/mixed integer/conversion/SFU/clock semantics are distinct from actual counter instruction class; no arithmetic mode inferred from benchmark name.','limitations':['Source inclusion only: first captured invocation may be warmup/setup; sample phase remains unproved.','No complete elapsed-ROI or complete launch census, multiplicity extrapolation, numerical recertification or causal bottleneck claim.','Alias nulling avoids repeated same-family candidates; it does not remove source calls or hide conditional branches.']}
   patches.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canon(s),'updates':updates})
  revised=[dict(s)for s in r['stages']]
  for p in patches:revised[p['stage_index']].update(p['updates'])
  matches=[]
  for ob in obs:
   name=ob['identity']['Kernel Name'];hits=[s['stage_id']for s in revised if s.get('semantic_family_in_primary_ROI')is True and s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],name)]
   assert len(hits)<=1,(b,m,name,hits)
   matches.append({'identity':ob['identity'],'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  if not any(x['matched_stage_ids']for x in matches):
   withheld.append({'benchmark':b,'model':m,'reason':'No positive saved included-family match; proposed model withheld','selected_observations':matches});continue
  parents[str(parent)]={'compressed_sha256':sha(pb),'uncompressed_sha256':sha(raw)}
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canon(r),'stage_patches':patches,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'ledger':{'path':str(lp),'sha256':sha(lb)},'manifest':file(mp),'plan':file(pp),'derivation':file(op)})
assert index_path.read_bytes()==index_bytes
assert sha(legacy_path.read_bytes())==legacy_authority['compressed_sha256']
assert sha(legacy_receipt_path.read_bytes())==legacy_authority['receipt']['sha256']
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'parent_INDEX':{'path':str(index_path),'sha256':sha(index_bytes)},'legacy_authority_exclusion_check':legacy_authority,'schema_version':1,'scope':'Private batch12 source-only enrichment sixteen remaining m-prefix then o-prefix named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
