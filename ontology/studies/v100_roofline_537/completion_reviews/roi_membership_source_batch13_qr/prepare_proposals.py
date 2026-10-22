import json,gzip,hashlib,re,datetime,fcntl,copy
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N='qem qkv qtclustering quantVLLM quicksort radixsort radixsort2 rainflow'.split()
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
why={'qem': 'Selected QuarticMinimumGPU and QuarticMinimumGPUStreams / qem_call! run two Float32 quartic-sourcebehavior stages, serial and four streams plus3-element remainder across1027 inputs, rep3. Fullhost timer includes20allocations, H2D, kernels, D2H, free/stream lifecycle; pinned fixture and independent Float64 sourcebehavior oracle outside timer. CUDAjl study_qr/study_solve selected, archived main launches uncalled. Native six sites all reachable through serial/full/remainder helpers, repeated family sites aliases without counter duplication. Preserve accepted sourcebehavior NOT globalquarticminimizer qualification; no rootaccuracy/application or sampledstream/phase claim.', 'qkv': 'Selected matmul_forward1 / run! computes naive Float32 dense input128x64 times weight192x64 transpose plus192bias into128x192output,16x16threads.100 launch+hostcompletion means; setup/warm and Int64 dyadic-dot oracle/readback outside. Native cuBLAS/add_bias and tiledkernel4 wrappers not called by selected driver; no vendor/tensor instruction claim or complete attention/transformer validation. CUDAjl hostrun! is not a devicekernel; selector narrowed to exact kernel_ callback visible in launch and savedfamily.', 'qtclustering': 'Selected complete quality-threshold compact-neighbor pipeline executes degree then candidatecluster/reduction/trim/update until source count<=1 termination. Float32 distances/cache, Int32 indices/cards and Int8 membership; five exact Val op states from source stage dispatcher. Sum separate launch+contextcompletion stage timers; hostcard readback, logicaloracle/fullmutableworkspace/guards and reset outside each stage timer. Source count<=1 residual semantics preserved, no external clustering optimum or complete elapsedhostROI claim. Same helper is warmed before measured trials, so sampled invocation phase remains unknown.', 'quantVLLM': 'Selected6fixturetypes/shapes x4static/dynamic symmetric/asymmetric modes quantize real FP16/BF16/FP32 storage toInt8 with Float32scales/Int32zero-points.24rep20kernel+completion means summed; reset/warm, fulloutput/tailguard and independent expectedfixture comparisons outside timers. Native study_native_kernels.cuh selected while original main.cu launchwrappers uncalled; CUDAjl study.jl invokes kernels directly, original main hostlaunch wrappers uncalled. Preserve accepted reduction/rounding/source qualifications; precision is source storage, not proof of half/tensor instruction issue, model inference accuracy or sampled mode/phase.', 'quicksort': 'Selected custom globalpartition/localfinish pipeline sorts24fixturecases of UInt32/Float32/Float64 with source pivot casts preserved. Preallocated fullhost H2D/hostworkqueue/kernelcompletion/D2H closure timed separately percase then summed. Setup/warm/resetguards and independent sortedpayload/recordguard/template checks outside. Native selected study_native pipeline reaches gqsort/lqsort; historical renamed CLI callsites excluded. No stability or arbitrarydomain guarantee and no causal framework/allocator attribution; sampled case/partitionphase unproved.', 'radixsort': 'Selected two UInt32length32768/65536 cases use8four-bit digits x6stages=48kernelcalls/fullsort, three stateful fullsorts mean percase summed. Exact localblocksort/offset/scan1/scan2/uniform/reorder stages, including JACC generic launch helper source enumeration of six concrete callback values. Reset/upload/warm and full sortedkeys+all scratchoracle outside timer; contextcompletion inside. Source family inclusion does not identify sampled digit/case or prove launchcensus/full elapsedROI; integeroperator has no usefulFPwork claim.', 'radixsort2': 'Selected argvfloat_pairs / nativeargs1,0 sorts1048576Float32keys withUInt32pairpayload,100reset-excluded launch+completion means. Julia source loops32one-bit localrank/shared1024scan/pairscatter steps; keyonly scatter branch uncalled for selectedonly=false. Full sortedkeys and pairpermutation/association checked outside timer; not stableties requirement. Native Thrust/CUB vendor families lack existing exactcallsite selector and remain held, no newvendorstage or genericordinalidentity. No applicationdataset or completecounterROI claim.', 'rainflow': 'Selected257variablelength Float64histories compute half/fullcycle counts/ranges/means withInt32offsets/stackindices;100launch+contextcompletion mean. Input/CPUlistoracle/warmup/fullresult+definedpadding readback outside timer, rawhistories unchanged. Native warm/timed callsites occupy samephysicalsource line with byte-identical stored selectors; native model held without ordinal assignment. Julia exact rainflow_eval! callback is reached through selected prepare closure; sourcefamily also warmed, sampledphase unproved. No fatigue-life/materialprediction or arbitraryrealdata certification.'}
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
   pattern=r'launch|factory|pipeline|study\(|study_|time_ns|chrono::|main\(|#define main|#undef main|kernel|case [4-7]:|c\.k==|mode=='
   for i,line in enumerate(lines):
    if re.search(pattern,line):picked.append({'line':i+1,'text':line})
   ev.append({'path':str(p),'sha256':e['sha256'],'line_evidence':picked})
   snips[str(p)]=e
  domain=W/'ontology/studies/julia_abstractions_537/domains'/f'{b}.json';d=json.loads(domain.read_text());scientific[b]={'binding':file(domain),'scientific_intent':d.get('scientific_intent'),'native_selected_configuration_id':r['configuration_id'],'native_scientific_intent_record_link':r.get('scientific_intent_source')}
  if b=='rainflow'and m=='cuda':
   assert r['stages'][0]['source_callsite']==r['stages'][1]['source_callsite']and r['stages'][0]['source_parameter_dispatch']==r['stages'][1]['source_parameter_dispatch']
   withheld_models.append({'benchmark':b,'model':m,'reason':'Distinct warmup/timed launches share identical stored source callsite and parameter selector on single line; no ordinal assignment or newstage.','plan':file(pp),'manifest':file(mp),'provenance':file(vp),'derivation':file(op),'original_record_canonical_sha256':canon(r),'bound_source_evidence':ev});continue
  patches=[];canonical_family={}
  for si,s in enumerate(r['stages']):
   if 'source_callsite'not in s:
    withheld.append({'benchmark':b,'model':m,'stage_id':s['stage_id'],'reason':'Existing stage has no exact source callsite/selector; no new stage or guessed identity supplied.'});continue
   path=Path(s['source_callsite']['path']).resolve();kind='timed_semantic_family';include=True
   candidates=s.get('source_parameter_dispatch',{}).get('callable_candidates',[])
   if b=='qem'and m=='cuda_jl'and path.name=='main.jl':kind='uncalled_original_host_wrapper';include=False
   if b=='qkv'and m=='cuda'and candidates!=['matmul_forward_kernel1']:kind='uncalled_native_mode2_or_mode4_wrapper';include=False
   if b=='quantVLLM'and((m=='cuda'and path.name=='main.cu')or(m=='cuda_jl'and path.name=='main.jl')):kind='uncalled_original_host_wrapper';include=False
   if b=='quicksort'and m=='cuda'and path==W/'src/quicksort-cuda/main.cu':kind='uncalled_original_CLI_wrapper';include=False
   if b=='radixsort2'and candidates==['scatter_keys_kernel']:kind='selected_float_pairs_excludes_keyonly_branch';include=False
   u={'semantic_family_in_primary_ROI':include,'ROI_membership_status':'source_timed_closure_inclusion_proved'if include else 'source_setup_exclusion_proved'}
   if not include:u.update(model_kernel_regex=None,regex_status='disabled_source_excluded_'+kind+'_not_runtime_phase_proof')
   if b=='qkv'and m=='cuda_jl':
    assert candidates==['run!','kernel!'];u.update(model_kernel_regex=r'\bkernel_(?=[(<])',regex_status='exact_selected_CUDA_launch_callable_kernel_not_host_run')
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
 out={'schema_version':1,'scope':'Private source-callpath membership batch13 eight selected completed-all4 q/r names, both legacy/indexed authorities and all12prior source/held names excluded; source12 currentparent pin; root scientific review/merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_stages':withheld,'withheld_models':withheld_models,'merging_policy':'Recheck initial parent compressed/raw SHA, canonical record/stage SHA and selected/source hashes. Immutable historical bytes and all phase/ROI/causal qualifiers preserved; root merger population constants need explicit batch-specific adaptation.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
 (O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n')
 proof={'sources':snips,'scientific_intent':scientific,'reviewed_names':N,'capture_bindings':selected,'scope':'Deduplicated exact full source bytes supporting source-only decisions; not new execution evidence'}
 (O/'proofpacket.json.gz').write_bytes(gzip.compress((json.dumps(proof,indent=2)+'\n').encode(),mtime=0))
 summary={'names':N,'model_records':len(proposals),'withheld_models':withheld_models,'stage_updates':sum(len(r['stage_patches'])for r in proposals),'inclusions':sum(st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'exclusions':sum(not st['updates']['semantic_family_in_primary_ROI']for r in proposals for st in r['stage_patches']),'withheld_vendor_placeholders':len(withheld),'sources':len(sources),'proposal_sha256':sha((O/'proposed_stage_patches.json').read_bytes()),'proofpacket_sha256':sha((O/'proofpacket.json.gz').read_bytes()),'metadata_read_lock':'LOCK_SH entire construction and stable final rehash','no_target_execution':True}
 (O/'unquoted_source_scope.json').write_text(json.dumps(unquoted,indent=2)+'\n')
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k!='withheld_models'},indent=2));print('WITHHELD',[(x['benchmark'],x['model'])for x in withheld_models])
