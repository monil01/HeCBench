import json,gzip,hashlib,re,datetime,fcntl,copy
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N='d2q9-bgk d3q19-bgk damage daphne dct8x8 ddbp debayer degrid dense-embedding deredundancy'.split()
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def file(p):p=Path(p);return {'path':str(p),'sha256':sha(p.read_bytes())}
def load(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def quoted(p):
 p=Path(p).resolve();b=p.read_bytes();t=b.decode();return {**file(p),'start_line':1,'end_line':t.count('\n')+1,'start_byte':0,'end_byte':len(b),'snippet':t}
why={
'd2q9-bgk':'Native Work::run invokes d2q9_bgk through its whole evolve loop, starts timer at t=999, fences at end; selected study main sums three resident cases. Julia main -> evolve(w,launch_step) explicitly excludes t=0:998, then invokes launch for t=999:steps-1 between start=time_ns and final context fence. Each launch_step binds the respective CUDA/KA/JACC step kernel. Warmup evolves the same function before trials. Historical original_cli_main is renamed and uncalled. Thus family occurs inside and outside timing; captured first invocation phase is unproved.',
'd3q19-bgk':'Selected native make_flag/find_wall execute once before timer; init_velocity_g belongs to reset invoked before each timed run. Only run lambda contains twenty collide_and_stream_g steps with population swap and per-step synchronization and is invoked after steady_clock start. Included historical_main is renamed and uncalled; its setup and collide callsites are dormant. Julia lbm_study takes explicitly bound lbm kernel callback, reset/copies/fills/readback outside timer; t=time_ns encloses run! twenty callback steps and swaps. Warmup run! repeats same family; no sampled phase proof.',
'damage':'Selected native launch lambda binds base damage_of_node, repeated ten times after start and before synchronized elapsed. Optimized and base launches in renamed original_hecbench_main are dormant. Each Julia factory explicitly binds damage kernel (KA instantiated handle or JACC callback); damage_study passes closure to OwnedStudy.run_trials!, whose launch loop is within timer. Input generation, allocation, reset, oracle and readback outside timer; same closure warms up.',
'daphne':'Selected native launch binds project_kernel, repeated twenty times between start and synchronized elapsed. Included original_daphne_main is renamed/uninvoked. CUDAjl study includes only main.jl text before function main(), then binds project_kernel to launch_julia!; KA/JACC launchers explicitly bind their named callbacks. daphne_study -> OwnedStudy.run_trials! encloses repeated launch! in timer; fixture/reference/readback outside timer. Warmup uses same callback.',
'dct8x8':'Native F::launch calls DCT8x8, whose dir conditional dispatches DCT8x8_kernel or IDCT8x8_kernel; selected study timer loops four fixtures times both directions with complete synchronization. Julia study_main likewise passes dirs666/777 through selected callback during timed eight-transform repeat loop. CUDAjl study includes main.jl without executing its guarded historical main; those historical forward/inverse launch callsites are dormant. KA named study_tile and JACC study_tile callback implement the dir branch. Both directions use one generic Julia family; sampled direction cannot be inferred. Vendor placeholder stages without callsite remain untouched.',
'ddbp':'Selected native main invokes pipeline(false) three repeats after timer for each of two cases. Pipeline includes projection padding, five map_boudaries calls, two img_integration calls, per-view rotation and per-slice map/interpolation/differentiation, then division, plus source-declared transfers and memset. Native corrected historical main is renamed and never called. Julia ddbp_study timer encloses pipeline!(...,nothing) three times; Val1..8 dispatch via CUDA/KA/JACC stage_kernel -> ddbp_lane! preserves all eight source branches. Untimed pipeline replay with capture callbacks follows timer; warmup uses same pipeline. Duplicate same-native-family callsites retain source inclusion but selectors alias canonical family; no repeated counter attribution or invocation-argument claim.',
'debayer':'Selected native per-case run lambda starts timer before source output memset plus malvar_he_cutler_demosaic loop, ends after device fence. Julia study::run follows same memset-plus-launch boundary; launch binds named CUDA kernel, instantiated KA kernel or JACC callback. Five fixtures/repeat means contribute source-defined timing; allocations, initial sentinels, fixture/oracle reads and readback outside timer. Warmup invokes same run; captured invocation phase unproved.',
'degrid':'Selected native run lambda calls degrid_kernel directly three repeats after start and before fence; host degrid wrapper in included kernels.cu is not called by this entrypoint. Julia study -> case! invokes selected launch three times between t=time_ns and context fence; CUDA direct degrid_cuda, KA DEGRID instance, JACC degrid_jacc callback. Two cases summed; exact rational reference and output reset/readback outside timer. Warmup uses case! too.',
'dense-embedding':'All four actual selected profile_target_command arrays end in k3; shared Julia dense_study reads that mode, factory resolves k3 and returns its explicit launch closure. Native study main parses argv[1]+1 and selects dense_esuhm3. k1/k2 branches are excluded for this capture. CUDAjl study includes only kernel definitions before function reference(), stripping historical main and exit; its historical three launch callsites are dormant. KA definitions k1/k2 are present but factory resolves only k3. JACC generic launch symbol resolves dense_k3_jacc! from same exact argument. OwnedStudy timer encloses ten callback repeats; reset/readback outside, warmup same closure.',
'deredundancy':'Selected native Device::pipeline and Julia pipeline! are invoked after synchronized host timer in run/runstudy and end with device synchronization. They include base/compress, fixture-k dispatch index4/5/6/7, cutoff, host compaction/sort and transfers, merge, representative-update loop and conditional make/magic/filter/align/clean callbacks. Identical pipeline runs in untimed tracing replay and warmup. Native historical main, corrected standalone main, and updateRepresentative allocation helper are not called by selected study main, which uses kernel_updateRepresentative directly. Every fixture-specific branch is a source family candidate only; no branch execution count or sample phase is inferred. JACC generic jlaunch passes explicit f callbacks; selector union enumerates only those source-bound kernel families. Vendor placeholders remain untouched.'}
rows=json.loads(gzip.decompress((O/'selected_review_records.json.gz').read_bytes()));proposals=[];parents={};sources={};selected=[];withheld=[];snips={};scientific={}
with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for row in rows:
  r=row['record'];b,m=r['benchmark'],r['model'];parent=Path(row['parent']);pb=parent.read_bytes();raw=gzip.decompress(pb);actual=json.loads(raw)['records'][row['index']];assert actual==r
  lp=S/'benchmarks'/f'{b}.json';ledger=json.loads(lp.read_text());state=ledger['models'][m];assert state['profiling_status']=='profile_process_completed'
  mp=W/state['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';op=mp.parent/'derived.json.gz';vp=mp.parent/'profile_provenance.json.gz';plan=load(pp);obs=load(op)['records'];prov=load(vp)
  assert plan['configuration_id']==r['configuration_id'] and plan['reference_result_sha256']==r['reference_result']['sha256']and plan['raw_config_sha256']==r['raw_config_sha256']
  for key in ['reference_result','raw_config']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
  bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()};bindings.update({str(Path(k).resolve()):v for k,v in prov.get('additional_current_profile_input_bindings',{}).items()})
  # Include all selected repository text bindings, retaining actual execution source instead of fabricating historical copies.
  paths={Path(p)for p in bindings if p.startswith(str(W/'src')+'/')and Path(p).suffix in ['.cu','.jl','.h','.hpp']}
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
   # Dormant callsites do not become selected families just because they share a symbol.
   if m=='cuda':
    if b in ['d2q9-bgk','damage','daphne','degrid']and si>=1:kind='dormant_legacy_entrypoint';include=False
    if b=='d3q19-bgk'and si!=3:kind='setup_before_timer'if si<3 else 'dormant_legacy_entrypoint';include=False
    if b=='ddbp'and si>=13:kind='dormant_legacy_entrypoint';include=False
    if b=='dense-embedding'and si!=2:kind='unselected_k1_k2_branch'if si<2 else 'dormant_legacy_entrypoint';include=False
    if b=='deredundancy'and si>=14:kind='dormant_legacy_entrypoint_or_uncalled_helper';include=False
   if m=='cuda_jl'and b=='dct8x8'and si in [1,2]:kind='dormant_legacy_entrypoint';include=False
   if m=='cuda_jl'and b=='dense-embedding'and si>0:kind='dormant_legacy_entrypoint';include=False
   if m=='ka'and b=='dense-embedding'and si<2:kind='unselected_k1_k2_branch';include=False
   u={'semantic_family_in_primary_ROI':include,'ROI_membership_status':'source_timed_closure_inclusion_proved'if include else'source_setup_exclusion_proved'}
   if not include:u.update(model_kernel_regex=None,regex_status='disabled_source_excluded_'+kind+'_not_runtime_phase_proof')
   if include and b=='daphne'and m=='cuda_jl':u.update(model_kernel_regex=r'\bproject_kernel_(?=[(<])',regex_status='source_explicit_launch_julia_project_kernel_alias_not_runtime_phase_proof')
   if include and b=='dense-embedding'and m=='cuda_jl':u.update(model_kernel_regex=r'\bdense_esuhm3_kernel_(?=[(<])',regex_status='source_k3_factory_dispatch_resolved_not_runtime_phase_proof')
   if include and b=='dense-embedding'and m=='jacc':u.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\n]*\bdense_k3_jacc_(?=[,<(])',regex_status='source_k3_factory_dispatch_resolved_not_runtime_phase_proof')
   if include and b=='deredundancy'and m=='jacc':u.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\n]*\b(?:base|compress|index[4567]|cutoff|merge|rep|make|clean|magic|filter|align)_kernel_(?=[,<(])',regex_status='source_explicit_callback_union_no_phase_or_multiplicity_proof')
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
  assert any(x['source_family_inclusion_only']for x in matches),('no saved source-family match',b,m)
  parents[str(parent)]={'compressed_sha256':sha(pb),'uncompressed_sha256':sha(raw)}
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':row['index'],'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canon(r),'stage_patches':patches,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'ledger':file(lp),'manifest':file(mp),'plan':file(pp),'derivation':file(op),'provenance':file(vp)})
 for p,e in sources.items():assert sha(Path(p).read_bytes())==e['sha256']
 for p,e in parents.items():assert sha(Path(p).read_bytes())==e['compressed_sha256']
 for r in selected:
  for k in ['ledger','manifest','plan','derivation','provenance']:assert sha(Path(r[k]['path']).read_bytes())==r[k]['sha256']
 out={'schema_version':1,'scope':'Private source-callpath membership batch4 ten alphabetically selected completed-all4 d* names; root scientific review and merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_stages':withheld,'withheld_models':[],'merging_policy':'Recheck initial parent compressed/raw SHA, canonical record/stage SHA and selected/source hashes. Immutable historical bytes and all phase/ROI/causal qualifiers preserved; root merger population constants need explicit batch-specific adaptation.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
 (O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n')
 proof={'sources':snips,'scientific_intent':scientific,'reviewed_names':N,'capture_bindings':selected,'scope':'Deduplicated exact full source bytes supporting source-only decisions; not new execution evidence'}
 (O/'proofpacket.json.gz').write_bytes(gzip.compress((json.dumps(proof,indent=2)+'\n').encode(),mtime=0))
 summary={'names':N,'model_records':len(proposals),'stage_updates':sum(len(r['stage_patches'])for r in proposals),'inclusions':sum(st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'exclusions':sum(not st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'withheld_vendor_placeholders':len(withheld),'sources':len(sources),'proposal_sha256':sha((O/'proposed_stage_patches.json').read_bytes()),'proofpacket_sha256':sha((O/'proofpacket.json.gz').read_bytes()),'metadata_read_lock':'LOCK_SH entire construction and stable final rehash','no_target_execution':True}
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
