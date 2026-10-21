import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set('gaussian gd ge-spmm geglu gelu glu'.split())
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
why={'gaussian': 'Selected native launch loops t0..126 with direct fan1 then fan2 inside synchronized timer; reset H2D/zeroes before timer, host BackSub and residual/oracle after. Native original gaussianElim main renamed/uninvoked. Julia factory explicitly returns same two-kernel forward-elimination loop (CUDAjl named functions, KA handles, JACC callbacks); gaussian_study passes operation! to OwnedStudy.run_trials!, whose reset precedes timer and readback/backsolve follows completion. Source Float32 nonpivoted elimination n128; reported timer covers forward elimination, not full linear solve or CPU backsolve. Device sampling phase unknown.', 'gd': 'Native train resets x before stamp then32iterations reset objective/l2/correct/gradient followed by compute and update inside synchronized timer. Included original main renamed/uninvoked. Julia training! has same zero-reset before time_ns and32iter fill! plus actual run! callback before sync/elapsed. Model wrappers explicitly bind compute and update; shared compute_body! has sparse Float32 dot/logistic exp/log and atomics; update includes weight decay and l2 reduction. CPU Float64 oracle and complete output readback outside. Memset/fill work IS inside timed training closure; unmatched fill kernel observations remain unassigned. This scoped sparse logistic-regression training state is not dataset prediction quality.', 'ge-spmm': 'Native call(method) chooses spmm_test1/2/3/4 and selected study visits all4methods,100repeats per synchronized timer after output memset. Julia study_main visits methods1:4 and invokes study_prepare-returned closure within study_loop! timer; actual callbacks explicit CUDA spmm_kernel!, KA/JACC portable_spmm!. CUDAjl study removes trailing exit(main(ARGS)) so legacy main candidates dormant. Source Float32 CSR sparse-by-dense multiplication at nr257/nc263/bc256. Native kernels use four distinct coarsening/mapping methods, Julia uses method-controlled column loop; source family membership does not establish identical lowering, full source method equivalence or causal resource effect. Allocation/warmup/oracle outside.', 'geglu': 'Native timed study repeats geglu_gpu wrapper, whose source calls explicitly templated geglu_kernel through dimension/forloop dispatch; wrapper is CALLED, unlike renamed historical main. Three supported widths1280/2560/5120,rows128 all within source timers. Julia study_main invokes prepare closure repeatedly then synchronize; explicit model callback geglu_kernel! binds Float32 erf gate calculation. Source x*gate*0.5*(1+erf(gate/sqrt2)); distinct from tanh-approx GELU and sigmoid GLU. Setup/warmup/Float64 scientific oracle/readback outside; scalar/SFU compiler instruction attribution still counter-specific.', 'gelu': 'All four captured commands explicitly select vec2,10repeats. Native timed launch chooses gelu_bias_loop, not gelu_bias_loop_base. Source kernel definitions are in hash-bound study_original_kernels.cuh; original main.cu candidates not called. Julia study sets mode from ARGS, factory explicit branch selects vec2 callback, and OwnedStudy times repeated operation after reset; CUDAjl study strips legacy exit(main(ARGS)). Baseline branch is excluded for selectedconfiguration, not unsupported/uncompiled globally. Source storage Float16; rounded half bias sum, Float32 tanh-polynomial evaluation and half projection; artifact Float32 widening outside timer. Repeated in-place operator, not a full neural-network workflow.', 'glu': 'Native study has first glu_kernel warmup before timer, second direct glu_kernel inside timed repeat, original historical main renamed/uninvoked. Julia study_main prepares seven source axis choices(2,0,1,2,0,1,2), invokes supplied closure in each synchronized repeat timer; explicit CUDA/KA/JACC glu_kernel! callback. Float32 x1*sigmoid(x2), dimension32 tensor, transfer/setup/warmup/oracle outside. Seven paths include repeated axis choices; aliases are not independent kernel families. Output scope is gated activation, not model quality.'}
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
  assert not plan.get('capture_overlay_provenance',{}).get('patches'),('requires independent overlay review',b,m)
  if m=='cuda':
   paths.add(base/('study_native.cu'if b in ['gaussian','gd','gelu']else'study_cuda.cu'))
   if b=='gelu':paths.add(base/'study_original_kernels.cuh')
  else:
   paths.add(base/'study_common.jl')
   for arg in plan['profile_target_command']:
    if str(arg).endswith('.jl')and Path(arg).is_file():paths.add(Path(arg).resolve())
   if b in ['gaussian','gelu']:paths.add(W/'src/_portable_study/OwnedStudy.jl')
  ev=[];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for p in sorted(paths):
   e=evidence(p)
   assert bindings.get(str(p.resolve()))==e['sha256'],('selected source binding gap',b,m,str(p))
   ev.append(e);sources[e['path']]=file(p)
  patches=[]
  for si,s in enumerate(r['stages']):
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
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
    updates.update(model_kernel_regex=r'_parallel_(?:for|reduce)_cuda[^\n]*\bgelu_vec2_jacc_(?=[,<(])',resolved_callable_candidates=['gelu_vec2_jacc!'],regex_status='exact_captured_vec2_mode_source_ternary_binding_baseline_not_selected')
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
out={'schema_version':1,'scope':'Private batch6 source-only enrichment six named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
