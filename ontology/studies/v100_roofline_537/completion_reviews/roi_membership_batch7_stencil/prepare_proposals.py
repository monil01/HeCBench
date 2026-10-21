import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set('gmm goulash gpp graphExecution gru'.split())
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
why={'gmm': 'Selected gmm_run starts host Clock before cluster() and subtracts explicitly measured pre/poststage observation intervals. Included controller_adapted.inc routes all7source stage calls through gmm_dispatch. Native dispatch switch invokes named native kernels; Julia common C-ABI callback wraps devicefields and explicitly calls launch_gmm(stage,...), whose branch maps to actual CUDA/KA/JACC kernels. The configured_native/cluster.cu direct launch candidates are NOT included; selected host_driver includes controller_adapted.inc, not cluster.cu. Source Float32 Gaussian-mixture EM/model-order convergence/merge controller and seven distinct GPU stage operators; tested d2/n992/capacity6. Guard allocation, transfers and CPUcontrol remain part of net timed pipeline; explicit observation intervals excluded. Source inclusion is net-timer callgraph proof, not contiguous elapsed ROI or captured phase classification.', 'goulash': 'Native selected study performs one gate warmup outside timer, then10updates from zero reset inside timer with sync before elapsed. Native included historical main renamed/uninvoked. Julia study(run!) invokes supplied explicit kernel!(gate,voltage,Int32(n)) callback in timed10loop; initial warmup/copy/reset and readback/oracle outside. Captured CUDAjl/JACC generic kernel signatures exactly match TWO Float64 arrays plusInt32; fullnames anchored, not host run! or bare dispatcher. KA saved signature has unexplained Float64 scalar in place of second devicearray and is separately withheld. Source polynomial/rational voltage-dependent gate target and exponential relaxation recurrence at4096fixedvoltages, not a whole cardiac model.', 'gpp': 'Native selected launch invokes solver once per each4case inside synchronized timer after outputreset; renamed original main dormant. Julia gpp_study invokes supplied launch(c) after reset/sync and before syncelapsed; CUDAjl study explicit solver_kernel!, KA/JACC explicit reduce_gpp! delegates gpp_thread! with Float64 split-complex products/divisions/atomic reductions. OriginalCUDAjl main guarded false whenstudyincluded. This is scoped electronic-structure generalized plasmon-pole complex reduction, not fullGW or electronic-spectrum calculation. Outputreset/fill and input/output/guard/oracle readbacks outside timer; saved fillkernel samples stayunassigned, no full scientific sample-phase proof.', 'graphExecution': 'Selected native segment copies pinned hostinput H2D,100times zeros partial and invokes reduce+reduceFinal, then D2H result. Exact segment is captured into graph before timing; each trial timer visits graphLaunch(executable) and direct segment separately with stream completion, sums both. Julia same captured segment closure invokes explicit launch_reductions callbacks and transfer/reset path; executable capture/instantiate outside timer, graphlaunch or CUDA.stream!(segment!,stream) inside. Source Float32 inputs accumulated into Float64 partial/final reductions; original renamed main candidates unused. Kernel family source membership applies to captured and ordinary segment paths, but first sampled invocation can be warmup/capture/other path; no attribution to timed graph-vs-stream segment or complete event census.', 'gru': 'Native selected study launches gru_cell_forward<half,float,int> inside100repeat synchronized timer, originalmainrenamedunused. Julia study_main invokes prepare-returned closure via study_loop! inside same timer; CUDAjl study removes legacy exit(main(ARGS)), KA/JACC explicitly bind portable_gru!. Source Float16 input/state storage, Float32 gate/sigmoid/tanh/intermediate arithmetic, half output plus all5saved-state planes; widening to Float32 artifact occurs outside timer. Inputs represent precomputed gate-affine values, not included matrixmultiplications/fullrecurrenttraining. Setup/warmup/reference and readback excluded; no tensor-instruction inference from half storage.'}
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
  if b=='goulash'and m=='ka':
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
  ev=[];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for p in sorted(paths):
   e=evidence(p)
   assert bindings.get(str(p.resolve()))==e['sha256'],('selected source binding gap',b,m,str(p))
   ev.append(e);sources[e['path']]=file(p)
  patches=[]
  for si,s in enumerate(r['stages']):
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
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
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'schema_version':1,'scope':'Private batch7 source-only enrichment five named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
