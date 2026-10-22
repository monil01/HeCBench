import json,gzip,hashlib,re,datetime,fcntl,copy
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N='s3d s8n sa sad sampling sc scan scan2'.split()
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def file(p):p=Path(p);return {'path':str(p),'sha256':sha(p.read_bytes())}
def load(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def quoted(p):
 p=Path(p).resolve();b=p.read_bytes();encoding='utf-8'
 try:t=b.decode(encoding)
 except UnicodeDecodeError:print('NON_UTF8_EXACT_SOURCE',str(p),sha(b));raise
 assert t.encode(encoding)==b
 return {**file(p),'snippet_encoding':encoding,'start_line':1,'end_line':t.count('\n')+1,'start_byte':0,'end_byte':len(b),'snippet':t}
why={'s3d': 'Selected argvf64 computes27chemical-rate/sourceformula stages on4096uniform cells inFloat64,100pipeline+contextcompletion mean. T1000/P1.0132e6/sourcecomposition andunitmolecularweight inputs are synthetic; sourceCPU508percell formula values, full WDOT/C/RF/RB/RKLOW/31EGarrays checked outside timer, not independentchemistry or coupledflow/timeintegration certification. All27native/Juliaentryfamilies source-reachable through selectedfactory; warmup uses same helpers so sampledphase/stage/cell unproved, no wholeROI/causal promotion.', 's8n': 'Selected two batches of520Int32coordinate points choose nearest1/2/4 neighbors in each8octants with strict squaredradius<512 and source index tieordering/selfdefault. Three100launch+completion means summed; input/oracle/warm/readback outside timer. Native selected studyhelper three calls included, originalCLIwrappers uncalled. CUDAjl/JACC source variablek is exactly ternary ranks1/2/4 concrete cube_select/cube_select_two/cube_select_four callbacks; selectedsource evidence enumerates three values instead of genericdispatcher. Allsourceoriginalprecision/layout preserved; no arbitrarynearestneighbor dataset/performance or sampled rankphase claim.', 'sa': 'Selected primarycases constructInt32suffixarray via source recursiveDC3 customentries plus unchanged ThrustCABI container/sort/scan bridge and hostrankfeedback. Timer includes perconstructionassign/H2D/recursive temporaryallocation+sort+scan+rankfeedback/free/D2H; outercasecontainerallocation/destruction and independent permutation/finite-tail lexorder certificate outside source timer. Source ninecustomoperations includes two keybitscallcontexts; repeatednativekeybits family aliasnull. CUDAjl hostsa_launch functions narrowed to exact launchedentry callbacks. Vendorplaceholderstages remain unresolved, no inventedCUBfamily/wholeROI/causal certificate or allinternaliterationoutputs archivedclaim.', 'sad': 'Selected fiveRGB UInt8image/template cases execute corrected accepted SADcompute thenInt32minimum andcount kernels,threepipeline+contextcompletion means summed,45launches sourcecount. Native selected study_native correctedcompute andsource min/count included; originalCLI launchwrappers excluded. Minimum20/count0 initialization, fullguards/input/reset/warm/fullindependentintegeroracle/readback outside timer. Preserve accepted corrected-addressing qualification; no untouched-originalnative repairclaim, physicalimagequality or sampledcase/phase/completeROI evidence.', 'sampling': 'Selected seven source exact-row plus sampled/complement-mask scatter pipelines useFloat32features/dataset,Int32samples andsourceUInt64seed. kernel_dataset / launchclosure+contextcompletion timed; reset/warm/fullbyte-mask/dataset/sourceinput/guards validation outside timer. Native duplicate canonical/noncanonical relativepath sites are same bound sourcebytecallsite aliases, no duplicatecounter attribution. No asserted simultaneouskernel overlap, statisticalestimatorquality/SHAPapplication accuracy or sampledcase/seed/phase/fullROI claim.', 'sc': 'Selected15static CPU/GPU Int32streamcompaction cases use4CPUworkers andone4block256thread GPUlaunch/case; hostworkerstart/contextcompletion/workerjoin inside full mixedcompute timer. Splitfraction sourceparameter cut, input/reset/padding/guards/count/flags/resultoracle/copymerge outside timer. Selectednative study run reaches StreamCompaction; originaldevice_sc hostwrapper uncalled. Julia exactdevicehelper included and vendor/CPUplaceholders not assigned GPUfamilies. Source CPU/GPU disjointownedbuffers/staticpartition preserved, no timedcounterwholeROI, replay/concurrencycertificate or causalframeworkspeed claim.', 'scan': 'Selected4096element exclusivepersegment scans aggregate5segmentsizes128..2048 x4integerwidths Int8/16/32/64 x2plain/bankpadding modes,three launchloop+completion mean/path. Native selectedstudy calls scan/scan_bcao; originalmain CLI wrappers uncalled. Julia exact sharedworkgroup scan_kernel implements source integer-width accumulation behavior; setup/warm/reset/readback/persegmentCPUoracle outside timer. No global4096prefix claim or exactreal-arithmetic guarantee from narrowinteger wrap behavior; sampledmode/type/phase and fullROI unknown.', 'scan2': 'Selected4096Float32 globalexclusiveprefix runsblockscan256 thentopscan16 andblockaddition256 pipeline,tenlaunchpipeline+completion mean. Selectednative bScan/pScan/bAddition sourcewrappers reach originaldevicefamilies; Julia prefixhelper two calls are onefamily sourcealiases plusaddition; no duplicatecounter attribution or sampleVal16/256 inference. Initialupload/warm/readback/Float64 CPU integer-valued prefixoracle outside timer. No arbitrarylength/precision or wholeROI/causal claim.'}
rows=json.loads(gzip.decompress((O/'selected_review_records.json.gz').read_bytes()));proposals=[];parents={};sources={};selected=[];withheld=[];withheld_models=[];snips={};scientific={};unquoted=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
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
   try:e=quoted(p)
   except UnicodeDecodeError:
    q=file(p);assert bindings.get(str(p))==q['sha256'];assert all(Path(st.get('source_callsite',{}).get('path','')).resolve()!=p for st in r['stages']);sources[str(p)]=q;unquoted.append({**q,'benchmark':b,'model':m,'reason':'Exact captured bytes retained by SHA; invalidUTF8 unquoted non-callsite dependency, no decode fallback or sourceclaim.'});continue
   assert bindings.get(str(p))==e['sha256'],('captured source mismatch',b,m,str(p));sources[str(p)]=file(p)
   # Compact exact source spans in the proposal; full snippets deduplicated in compressed proof packet.
   lines=e['snippet'].splitlines();picked=[]
   pattern=r'launch|factory|pipeline|study\(|study_|time_ns|chrono::|main\(|#define main|#undef main|kernel|ranks==|for ranks|sa_launch|CUDA.*synchronize|case [4-7]:|c\.k==|mode=='
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
   candidates=s.get('source_parameter_dispatch',{}).get('callable_candidates',[])
   if b in ['s8n','sad','scan']and m=='cuda'and path.name=='main.cu':kind='uncalled_original_CLI_wrapper';include=False
   if b=='s8n'and m=='cuda_jl'and path.name=='main.jl':kind='uncalled_original_host_launch_wrapper';include=False
   if b=='sc'and m=='cuda'and path.name=='device_sc.cu':kind='uncalled_original_host_launch_wrapper';include=False
   if b=='s8n'and m in ['cuda_jl','jacc']and not candidates:kind='selected_three_value_callable_dispatch_proved'
   u={'semantic_family_in_primary_ROI':include,'ROI_membership_status':'source_timed_closure_inclusion_proved'if include else 'source_setup_exclusion_proved'}
   if not include:u.update(model_kernel_regex=None,regex_status='disabled_source_excluded_'+kind+'_not_runtime_phase_proof')
   if b=='s8n'and m in ['cuda_jl','jacc']and not candidates:
    callbacks=['cube_select_kernel_','cube_select_two_kernel_','cube_select_four_kernel_']
    u.update(model_kernel_regex='|'.join((r'\b'+v+r'(?=[(<])'if m=='cuda_jl'else r'_parallel_(?:for|reduce)_cuda[^\n]*\b'+v+r'(?=[,<(])')for v in callbacks),regex_status='exact_source_ternary_ranks_1_2_4_callable_values_not_generic_dispatcher')
   if b=='sa'and m=='cuda_jl':
    concrete=[v for v in candidates if v!='sa_launch'];assert len(concrete)==1
    u.update(model_kernel_regex=r'\b'+re.escape(concrete[0].replace('!','_'))+r'(?=[(<])',regex_status='exact_selected_CUDA_entry_not_host_sa_launch')
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
 out={'schema_version':1,'scope':'Private source-callpath membership batch15 eight selected completed-all4 s/t names, dualauthorities/all12adopted/knownholds/13and14reservations excluded; saxpy-ompt exactcanonicalCIDnull upfrontheld, scan2 replacement; source12currentparents pinned, rootreviewrequired','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_stages':withheld,'withheld_models':withheld_models,'merging_policy':'Recheck initial parent compressed/raw SHA, canonical record/stage SHA and selected/source hashes. Immutable historical bytes and all phase/ROI/causal qualifiers preserved; root merger population constants need explicit batch-specific adaptation.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
 (O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n')
 proof={'sources':snips,'scientific_intent':scientific,'reviewed_names':N,'capture_bindings':selected,'scope':'Deduplicated exact full source bytes supporting source-only decisions; not new execution evidence'}
 (O/'proofpacket.json.gz').write_bytes(gzip.compress((json.dumps(proof,indent=2)+'\n').encode(),mtime=0))
 summary={'names':N,'model_records':len(proposals),'withheld_models':withheld_models,'stage_updates':sum(len(r['stage_patches'])for r in proposals),'inclusions':sum(st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'exclusions':sum(not st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'withheld_vendor_placeholders':len(withheld),'sources':len(sources),'proposal_sha256':sha((O/'proposed_stage_patches.json').read_bytes()),'proofpacket_sha256':sha((O/'proofpacket.json.gz').read_bytes()),'metadata_read_lock':'LOCK_SH entire construction and stable final rehash','no_target_execution':True}
 (O/'unquoted_source_scope.json').write_text(json.dumps(unquoted,indent=2)+'\n')
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='withheld_models'},indent=2));print('WITHHELD',[(x['benchmark'],x['model'])for x in withheld_models])
