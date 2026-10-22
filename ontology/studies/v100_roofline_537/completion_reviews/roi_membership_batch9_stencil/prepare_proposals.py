import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set('lavaMD layernorm layout lfib4 log2 logprob'.split())
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
why={'lavaMD': 'Native selected run(Case) directly invokes md per threecases within synchronized singlelaunch timer after resetzero; originalmain launch candidate notcalled. Julia study invokes explicit launch(meta,rv,q,fv,alpha,nb) between time_ns and synchronize; CUDAjl/JACC md_kernel! delegates md_body!, KA namedkernel includes same tiledpairloop. Source Float32 Gaussian pairinteraction/forces, packed box/neighbor metadata and StudyFour storage,128threads/block and shared900Float32 tile; outputs accumulatedfromzero. Guard/inputecho/weightedforcebalance/oraclechecks aftertimer, allocation/initialwarmup before. Namedsingleforceoperator cases, not a timeintegrated molecular trajectory; serialized Float64 is encoding not compute precision.', 'layernorm': 'Native selected study lambda invokes layernorm_forward dispatcher inside10repeat timer for6fixtures×3algorithms×6blockwidths; originalsource wrappers0/1/2 each invoke namedkernel then device_synchronize. Kernel0 source C%4 branch chooses templateUNROLL4 or1, exact selectors disambiguated. Julia layernorm_study builds explicit callback peralgorithm/blockwidth and calls10times with synchronize aftereach within timer; algorithm0 uses ln0_kernel!,1/2 use lnwarp_kernel! with Valmode. Factory bindings explicit forCUDAjl launchfalse handles, KA namedhandles, JACC callbacks. Source Float32 mean/variance/reciprocalstd affine normalization; vector4load mapping distinct from arithmeticprecision. Guardreset/warmup/inputecho/fulloracle after/beforetimers; no fullnetworktraining claim.', 'layout': 'Captured all4commands explicitly soa,10. Native selected launch invokes SoAKernel<TREE_SIZE> inside10repeat timer; AoSbranch unselected and originalmain loops dormant. Julia layout_study selects soa fixture, factory explicit branch binds soa_kernel!/soa_ka!/soa_jacc!, then OwnedStudy times10launches after outputreset; CUDAjl study sets includeonly guard before including oldmain. Source 4096rows×4096integerentries, UInt32wrap sums projectedInt32; not scalarFPwork. No AoS/SoA performance comparison inferred from this oneconfiguration. Setup/transfers/oracle outside, firstsamplephase unknown.', 'lfib4': 'Native timed pipeline launches firstColGPU and colYGPU on two streams, joins both, then launches lastEntGPU and colsGPU followed by devicecompletion; resetseed/zero outsidetimer. Renamed originalmain directlaunchaliases dormant. Julia pipeline! uses explicit first_column!/column_y!/last_entries!/columns! callbacks throughCUDA/KA/JACC with same twostream joins and finalcompletion; main timercalls fullpipeline. Source UInt32 fourlag additiveFibonacci recurrence modulo2^32, n16384,r16, fixedseedstate; no scalarFP/quality/MonteCarlo guarantee. Allocation/JACCsharedworkspaces/warmup/oracle/readback outside. Four sourcefamilies and source dependencyedges do not prove realizedstreamoverlap, throughput bottleneck or fullprofilerphase.', 'log2': 'Native selected suite directly invokes compute_log for19approximationdepths each17repeats within timer, compiled with exact -DSTUDY_FIX_PRECISION=1 selecting study_corrected_kernel.h. Headerwrapper run_test candidates in original/corrected headers are not called by study and aliases disabled. Julia suite! invokes factorycallback log_cuda!/log_ka!/log_jacc! with explicit row,p for19depths and17repeats before synchronizedelapsed. Source Float32 repeatedsquare binarylogapproximation plus UInt64denominator shift (recorded correctedreference convention), NOT19hardwareprecision formats. Reset/inputecho/oracleaccuracy budget andguardchecks outside; alloldreferencequalifications preserved.', 'logprob': 'Native selected launch invokes log_probs_kernel<float> followed by accumulate_log_probs within10repeat synchronizedtimer; originalincludedmain renamed/dormant. Julia studyfactory closure explicitly binds log_probs_kernel!/accumulate_kernel!, KA namedpair/JACC callbacks, and OwnedStudy times tenpaircalls afterreset. CUDAjl include_string only defines originaldeviceprefix before CPUroutine, so oldhostlaunchaliases uninvoked. Float32 stabilized logsoftmax of logits for providedtokenIDs plus sequence sum; source exp/log/reductions andInt32ids/lengths, batch64,maxlen32,vocab256. Setup/warmup/reset/readback/oracle outsidetimer; operator scope is scoreevaluation, not inference/token generation or language-modelquality.'}
lock=Path('/tmp/hecbench-roofline-metadata.lock').open('r');fcntl.flock(lock,fcntl.LOCK_SH)
proposals=[];sources={};parents={};selected=[];withheld=[]
legacy_path=S/'ROI_REVIEW.json.gz';legacy_packed=legacy_path.read_bytes();legacy_raw=gzip.decompress(legacy_packed);legacy_doc=json.loads(legacy_raw);legacy_receipt_path=S/'ROI_REVIEW_RECEIPT.json';legacy_receipt=json.loads(legacy_receipt_path.read_bytes())
assert sha(legacy_packed)==legacy_receipt['sha256'] and sha(legacy_raw)==legacy_receipt['uncompressed_sha256']
assert not any(r['benchmark']in N for r in legacy_doc['records']), 'Legacy authority overlaps proposed names'
legacy_authority={'path':str(legacy_path),'compressed_sha256':sha(legacy_packed),'uncompressed_sha256':sha(legacy_raw),'receipt':file(legacy_receipt_path),'none_of_selected_names_present':True}

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
  base=W/'src'/f'{b}-julia'; paths={Path(s['source_callsite']['path']).resolve() for s in r['stages'] if s.get('source_callsite')}
  assert not any(str(stage.get('ROI_membership_status','')).startswith('source_')for stage in r['stages']),('Already source reviewed',b,m)
  overlays=[]
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
  ev=[];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for p in sorted(paths):
   e=evidence(p)
   assert bindings.get(str(p.resolve()))==e['sha256'],('selected source binding gap',b,m,str(p))
   ev.append(e);sources[e['path']]=file(p)
  patches=[]
  for si,s in enumerate(r['stages']):
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
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
    updates.update(model_kernel_regex=r'\blayernorm_forward_kernel0<\(int\)'+str(template)+r'>',regex_status='exact_source_Cmod4_UNROLL4_vs1_template_branch')
   if b=='layout'and m=='jacc':
    updates.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\n]*\bsoa_jacc_(?=[,<(])',resolved_callable_candidates=['soa_jacc!'],regex_status='exact_captured_soa_mode_source_ternary_dispatch')
   updates['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':[{'path':e['path'],'sha256':e['sha256'],'timer_lines':[{'line':e['start_line']+i,'text':line}for i,line in enumerate(e['snippet'].splitlines())if any(v in line for v in ['time_ns()', 'steady_clock::now()', 'Clock::now()', 'excluded+=', 'run_trials!(', 'bm_study(factory)', 'exit(study_main(', 'bezier_study(', 'box_study(factory)'])]}for e in ev],'captured_overlay_source_evidence':overlays,'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'selected_environment_overrides':plan['environment_overrides'],'precision_scope':'Source types/operations/Val branches are in exact evidence; selectedenvironment retained. Float32/Float64/mixed integer/conversion/SFU/clock semantics are distinct from actual counter instruction class; no arithmetic mode inferred from benchmark name.','limitations':['Source inclusion only: first captured invocation may be warmup/setup; sample phase remains unproved.','No complete elapsed-ROI or complete launch census, multiplicity extrapolation, numerical recertification or causal bottleneck claim.','Alias nulling avoids repeated same-family candidates; it does not remove source calls or hide conditional branches.']}
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
assert sha(legacy_path.read_bytes())==legacy_authority['compressed_sha256']
assert sha(legacy_receipt_path.read_bytes())==legacy_authority['receipt']['sha256']
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'legacy_authority_exclusion_check':legacy_authority,'schema_version':1,'scope':'Private batch9 source-only enrichment six named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
