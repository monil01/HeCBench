import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set(['merkle', 'metropolis', 'michalewicz', 'miniFE', 'miniWeather', 'minibude', 'minimap2', 'minimod', 'minisweep', 'minkowski', 'minmax', 'mis', 'mixbench', 'mlp', 'mmcsf', 'mnist'])
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
why={'merkle': 'Julia explicit launch calls phase0 then phase1 across all tree levels with context completion inside threecase fullmerklize timer. UInt64 Goldilocks finitefield RescuePrime arithmetic, not ordinary scalar floating point. Native candidate references original merkle_tree.cu but actualselected correctedinc changes leafstride; without exact candidate-source alias adjudication native withheld rather than silently attributing original source. Qualified correction remains; no external security or production-strength claim. Resets/readback/oracle outside.', 'metropolis': 'Selected corrected kernel_metropolis and explicit Julia launch perform two alternating block tilings per source sweep, synchronize each, inside threecase pipeline timer. Source Ising32cubed lattice Int32spins/field, UInt64LCG/RNG state, Float32coupling/temperature and exp acceptance; RNG and physics distinct from pureinteger work. Actualinputreadbacks/reset before, complete spin/RNG checks after. Finite named sweeps, not equilibration/thermodynamic convergence or arbitrary distributedsimulation.', 'michalewicz': 'Only seeded eval/minimum reduction/mx_reduce repeated100times enters twelvepath timers; capture_objective/mx_capture warmup and fullobjective diagnostic launches outside timer explicitly excluded. First eval warmup and original main uncalled. Float32 sin/pow objective plus integerCAS atomic minimum, dimensions2/5/10 and257/4099points; source seedzero retained. Captured min is constrained supplied points/seed, not proof of global optimum.', 'miniFE': 'Common mini_run times source Q1 assembly and full CG controller for cells2/16/128; added input, reduction-history and final independent observations subtracted. Julia CABI op packet dispatch explicitly maps vector_entry,matvec_entry,dot_entry,final_entry to authored kernels. Float64 sparsematrix/vector/reduction arithmetic with integerCSR. Native directvector branch reachability not independently resolved here; those candidates unchanged/withheld. Source assembly templatedTkern1D andCSRmatvec/dot/finalreduce paths source-bound within called driver. Allocation/transfers/CPU CG scalars and assembly part of nettimer, not device-only. Conditional reference/true residual and boundary checks retain existing scope.', 'miniWeather': 'Native study integrate calls original perform_timestep; Julia integrate! executes same splitRK three-substep axes with MPIself halo exchanges inside timer. Pack/unpack,flux/tendency,zhalo and fluidupdate families included; mass/energy acc_mass_te diagnostics before/after timer excluded. Native update_state_x injection branch unselected by activated DATA_SPEC_THERMAL. Float64 state/flux and integercoordinates, MPIhostcopies/request completion included. Threefinite namedcases/schedules and mass control, not distributedmanyrank or validatedlongtermweatherprediction.', 'minibude': 'Selected Work.run/nativefasten_main and explicit Julia modelcallback run sixcase source dockingenergy operators sevenrepeats inside percase synchronizedtimer summed. Float32 atom/forcefield/pose parameters and trig/transcendental energy arithmetic, source AoS/sharedforcefield staging retained. Setup/reset/fullenergyoracle and actualreadbacks outside,300? no warmupcount inferred beyond actualsource. Named forcefield scoring, not experimental binding affinity or full docking search; old standalone originalmain launches dormant.', 'minimap2': 'Selected native device_chain_tiled and Julia launch! callbacks execute every source-scheduled batched64lane circularDP chaining launch with finalwholedevice completion inside timer. UInt64packedanchor/control, Int32scores/parents plus Float32gap penalty term/avgspan conversion; not pureFP or entirelyinteger. Scheduler/allocation/inputcapture and complete DPparent/max/return checks outside. Named source chaining operator, not complete readmapping/alignment/assembly accuracy or actualsequencingthroughput.', 'minimod': 'Source launch loops allseven spatialregions per timestep: one interior and six PML regions then pointsource, buffer parityswap. Julia wave callback uses Val(PML) plus sourcekernel explicitly. Alllaunches and finalcompletion inside sumthreecase timer; resets/inputecho and fullstate CPUoracle outside. Float32 highorderwave/PML plus integerindexing, shared4096Float32tiles; scientific region boundary and source controls preserved. Named24cubed domain, not general geophysicalmodel validation.', 'minisweep': 'Factory launches three faceinit families then persistentwavefront inside two-case repeat timer, which also includes outputmemset and completion; source native same chain. Float64 angular transport-like sweep, integerwavefront flags/dependencies; initializer names here denote timed scientific boundaryface operators, not outside setup. Periteration fullreadback appended after timer; guard/oracle checks outside. No inferred overlap/runtimeblockorder or full elapsedROI counteraggregation.', 'minkowski': 'Native timedrepeat launch and Julia run! explicitcallback compute rectangular matrix pairdistances for p1..4, summed fourkernelmeans; firstnativewarmup call and originalmain excluded. Float32 abs difference, power and root accumulation across128features,64x128points. p is distance-order parameter, not hardwareprecision; inputreset/readback/CPUoracle outside.', 'minmax': 'OnlyKAparent has directsource stage; extrema_stage with Val FIRST/MODE iterates Float32 squared-distance min/max and Int32stabletieindices, recursively reduces and returns hostresults inside sumtwooperation repeat timer. Native/CUDAjl/JACC library/placeholderstages held without emittedfamily inference. Existing source correctness not re-certified; no inference noSMwork from authoredkernelabsence.', 'mis': 'Source seven graph cases time initpriority then async findmins three-repeat kernels pluscompletion; warmup and priority-capture init replay aftertimedfinal are outside. Explicit Julia Valdispatcher maps init/findmins samefamilies; source callback can run in both main and observation contexts so actualsamplephase unknown. UInt8status/priority and Int32CSR/graph arithmetic. Independent edge/maximality checks outside; maximal independent set not maximum-cardinality optimum.', 'mixbench': 'Native selected repeatloop and explicit Julia launchbenchmark_kernel! or KA/JACC callbacks time fixed computeiterations16 recurrence with datafeedback,20repeats and completion. Firstnativewarmup and originalmain dormant. Float32 arithmetic plus sparse addressedpayload work; recurrence/state initialreset and fullsourceoracle outside. Sourceunroll work not measured executedFLOP/peakperformance claim.', 'mlp': 'KA/JACC explicit dense_tile factory loops three784to512to512to10 layers, selectedbias/ReLUflags, inside tenforwardrepeat timer;300sourcewarmups before. Float32tiled GEMM/activation/bias, no tensorinstruction inference from operator name. Native/CUDAjl placeholders vendor/library only withheld; actual vendorSMwork remains unknown. Fixtureforward accuracy not trainednetwork/generalclassificationquality.', 'mmcsf': 'Selected qualified secondary_fp64_atomic full mm_run invokes four complete CSFsourcecases/allmodes, callbackop0..6 explicitlydispatching seven authoredkernels. Native configureddispatch uses same7functions. Source Float32 contribution expression converted after completion into FP64 atomicoutput; per-modeDoubleOut allocation/zeroing/D2HprojectiontoFloat32/H2D/free stay inside nettimer, added Observe input/oracle intervals excluded. Ranks<=32 scope and original strictblocked/referenceprecision qualification preserved; not originalpureFP32 or ALSfeedback. CABI aliases/pointer8 capacity and readonlyinputs actualsourcebound.', 'mnist': 'Instrumented native learn(2)/test and Julia learn!/test! wholepipeline timer includes selected100train20test datasetworkflow, CNNforward/backprop/update/hostcontrol/memsets; contextcompletion before observer pauses stays timed, extra statecaptures/oracle IO excluded by Pause/paused. Native source main genuine invoked via driver except originalmain entrywrapper.19Float32 devicefamilies with integerindices; repeatedapplystep/applygrad aliascalls retained without duplicatecounterattribution. Parameterreset/datasetread/physicalredzones outside. Bounded twoepoch numericalworkflow, not fullMNIST benchmarktraining/clinicalprediction/quality or pureGPUtime.'}
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
   if p.suffix in ('.jl','.cu','.cuh','.h','.hpp','.cpp','.c') and (str(base)+'/' in str(p) or '/src/'+b+'-cuda/' in str(p) or str(p).endswith('/src/_portable_study/OwnedStudy.jl')):
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
   if not s.get('source_callsite'):continue # no vendor/API-name-to-emitted-kernel promotion
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   if b=='merkle'and m=='cuda':continue # corrected source not canonicalcandidate path
   if b=='miniFE'and m=='cuda'and si not in [0,1,11,12]:continue # unresolved vectorbranches/duplicateoverloads
   excluded=None
   if m=='cuda':
    if b=='michalewicz':excluded='untimed_warmup'if si<2 else'untimed_full_objective_capture'if si==3 else'uninvoked_legacy_main'if si>=4 else None
    elif b=='miniWeather':excluded='untimed_mass_energy_diagnostic'if si in [0,9]else'argv_thermal_disables_injection_branch'if si==7 else None
    elif b=='minibude':excluded='uninvoked_legacy_main'if si>=1 else None
    elif b=='minisweep':excluded='uninvoked_legacy_main'if si>=4 else None
    elif b=='minkowski':excluded='untimed_warmup'if si==0 else'uninvoked_legacy_main'if si>=2 else None
    elif b=='mis':excluded='untimed_warmup'if si<2 else'untimed_priority_initialization_replay'if si==4 else'uninvoked_legacy_main'if si>=5 else None
    elif b=='mixbench':excluded='untimed_warmup'if si==0 else'uninvoked_legacy_main'if si>=2 else None
   else:
    if b=='michalewicz'and si==1:excluded='untimed_full_objective_capture'
    if b=='miniWeather'and si==8:excluded='untimed_mass_energy_diagnostic'
   kind=excluded or 'timed_semantic_family';included=excluded is None
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_untimed_audit_exclusion_proved'if excluded in ['untimed_full_objective_capture','untimed_mass_energy_diagnostic','untimed_priority_initialization_replay']else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_'+excluded)
   if included and b=='mixbench'and m=='cuda_jl':updates.update(model_kernel_regex=r'\bbenchmark_kernel_(?=[(<])',resolved_callable_candidates=['benchmark_kernel!'],regex_status='exact_selectedstudy_authored_benchmarkcallback_not_hostlaunchname')
   regex=updates.get('model_kernel_regex',s.get('model_kernel_regex'))
   if included and regex:
    if regex in canonical_families:updates.update(model_kernel_regex=None,alias_of_stage_id=canonical_families[regex],regex_status='same_named_source_family_callsite_alias_no_duplicate_counterattribution')
    else:canonical_families[regex]=s['stage_id']
   updates['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':[{'path':e['path'],'sha256':e['sha256'],'timer_lines':[{'line':e['start_line']+i,'text':line}for i,line in enumerate(e['snippet'].splitlines())if any(v in line for v in ['time_ns()', 'steady_clock::now()', 'Clock::now()', 'excluded+=', 'run_trials!(', 'bm_study(factory)', 'exit(study_main(', 'bezier_study(', 'box_study(factory)'])]}for e in ev],'captured_overlay_source_evidence':overlays,'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'selected_environment_overrides':plan['environment_overrides'],'precision_scope':'Source types/operations/Val branches are in exact evidence; selectedenvironment retained. Float32/Float64/mixed integer/conversion/SFU/clock semantics are distinct from actual counter instruction class; no arithmetic mode inferred from benchmark name.','limitations':['Source inclusion only: first captured invocation may be warmup/setup; sample phase remains unproved.','No complete elapsed-ROI or complete launch census, multiplicity extrapolation, numerical recertification or causal bottleneck claim.','Alias nulling avoids repeated same-family candidates; it does not remove source calls or hide conditional branches.']}
   patches.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canon(s),'updates':updates})
  revised=[dict(s)for s in r['stages']]
  for p in patches:revised[p['stage_index']].update(p['updates']);matches=[]
  for ob in obs:
   name=ob['identity']['Kernel Name'];hits=[s['stage_id']for s in revised if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],name)]
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
out={'parent_INDEX':{'path':str(index_path),'sha256':sha(index_bytes)},'legacy_authority_exclusion_check':legacy_authority,'schema_version':1,'scope':'Private batch11 source-only enrichment sixteen remaining m-prefix named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
