import json,gzip,hashlib,re,datetime,fcntl
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N=set(['mandelbrot', 'marchingCubes', 'mask', 'match', 'matern', 'matrix-rotate', 'matrixT', 'maxpool3d', 'mcmd', 'md', 'mdh', 'meanshift', 'medianfilter', 'memtest', 'merge', 'mergeVS'])
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
why={'mandelbrot': 'Selected study calls mandel through explicit closure inside one synchronized timer per named image; header original-main launch is dormant. Julia explicit factory closure enters same per-image launch+completion timer. Source Float32 complex recurrence with explicit fusion convention and Int32 escape counts, not image quality or continuum fractal validation. Input/parameter capture, setup and full oracle after/before timer. First captured invocation may be warmup.', 'marchingCubes': 'Only generatingTriangles/mc_stage5 closure is timed. First four stages minmax/active-cube compaction are invoked in preprocess! and separately reported untimed_preprocessing_seconds before start. Native study similarly preprocesses before timed generation closure; dormant original main aliases excluded. Julia factory maps kernel1..5 explicitly to corresponding ops. Source Float32 field/interpolation, UInt table/index/count and partial coordinate/checksum output; full mesh arrays not produced. No all-pipeline timed-ROI claim; profiling firstperfamily may sample preprocessing.', 'mask': 'Native six switch cases and Julia Val(mode) dispatcher called for six mask operators across two fixtures in repeat timer. Selected sequence/window/upper/lower/diagonal predicates operate Int32 values/indices, not scalarFP. Warmup and CPU independent predicate/oracle checks outside. Legacy original main host loops dormant. Generic callback maps to explicit mask kernel with mode; vendor placeholder is not proof of emitted library family.', 'match': 'Native selected switch directly launches ten matching kernels per two cases inside three-repeat timer. Julia explicit k-dispatch maps same ten callbacks/KAhandles; repeated dot-product argmax and tie-order fixture semantics source-bound. Float32 input/product/reduction with Int32 winning IDs, not general nearest-neighbor or arbitrary tie optimality claim. Upload/reset and score/index oracle outside; legacy original/study_source standalone main dormant.', 'matern': 'Selected native and Julia invoke matern kernel across two weight modes and seven scales in five-repeat synchronized timer; first direct native invocation warmup outside. Source Float32 distance/SFU weighted kernel sum with Float64 independent oracle, positive-weight monotonicity/bounds tested outside. Not covariance-matrix SPD/general statistical model validation; original main diagnostics uncalled.', 'matrix-rotate': 'Selected N1024/repeats10: native first direct launch warmup excluded; repeat-loop in-place rotate launch timed. Julia explicit rotate callback in common repeat timer. Float32 storage moved by integer ring/cycle indexing; no ordinary FP arithmetic implied by stored format. Input reset, CPU rotation oracle and readback outside timer.', 'matrixT': 'Native study uses nine function-pointer entries including copies, partial transposes and complete transposes, each inside own repeat timer summed; unrelated original main launch candidate alone insufficient for inclusion. Julia factory tuple indexkind1..9 binds explicit callbacks/KAhandles/JACCmode dispatch in same nine per-operation timers. Float32 payload movement with integer indexing; source vector/shared layouts do not imply FP arithmetic. Original host main/benchmark wrappers uncalled; no full-image numerical method claimed.', 'maxpool3d': 'Native study invokes maxpool3d through closure inside ten-repeat timer; original standalone main dormant. Julia prepare binds explicit pool callback and OwnedStudy run_trials times same repeats after reset. Named256x256x16 shape with2x2x2 max selection, Float32 storage/comparisons plus integer offsets. Warmup/reset/transfers/CPUoracle outside; not full neural-network training.', 'mcmd': 'Common native mcmd_run host controller calls force_kernel and authored Julia CABI callback into explicit mcmd_launch. Native force_dispatch launches calculateForceKernel; archived original force_kernel wrapper uncalled. Timer belongs full original11force-call MC/MD workflow including CPU controller/allocation/H2D/D2H/free; instrumented input/output and host-state observer durations subtracted, not GPU completion. Float64 pform1 LJ+Ewald finite fixture; no arbitrary potential modes or full experimental physics certification. Generic vendor candidate not emitted-kernel proof.', 'md': 'Selected native/Julia run three129-particle force-only fixtures20times inside synchronized timer; warmup direct native call outside. Float32 cutoff neighbor-list force with source LJ constants, independent Float64 pairwise/potential-gradient and symmetry checks outside. Coincident/exact-cutoff controls preserved; no integrated MD trajectory. Original host aliases uncalled; vendor placeholder unresolved.', 'mdh': 'Selected native invokes three vectorized potential kernels and Julia run!Val modes0/1/2 inside per-mode10-repeat timers summed. Float32 four-lane storage/accumulation, shared atom tiles, screened-charge exp/sqrt; vector payload not FP16/tensor proof. Atom129/grid1024 fixture setup and Float64 screened-potential oracle outside. Original main aliases dormant.', 'meanshift': 'Captured argv tiled selects only tiled native/KA/JACC callback branch; base branch explicitly excluded. Factory executes1000iteration ping-pong launches with periteration completion, timed total divided1000; reset and converged-state oracle outside. Float32 Gaussian3D mean shift with distance threshold60 and denominator32,10000points. No sampled iteration/convergence speed or causal cache/shared claim.', 'medianfilter': 'Selected ckMedian launch and Julia explicit callback execute100repeat timer after warmup. Source packed UInt32 RGB bytes, eight binary-search steps and zero-extended3x3 neighborhood; true rank median check allows source one-byte approximation difference, alpha0 contract retained. Integer/bitwise image operator, not Float32 despite historical labels; oracle/readback outside.', 'memtest': 'Source timed pipeline init/move/check executes three repeats with per-stage completion; separate intentional fault/check probe before primary timer is excluded. Julia Symbol dispatch explicitly picks init!,move!,check!, rather than generic JACC f identity by name alone. UInt32 patterns/movement plus UInt64 address/error bookkeeping,2MiB scoped test5; injected-fault and fullpayload checks outside. Not all memory tests or hardware reliability certificate.', 'merge': 'Captured mode float64; selected native diagonal-path then window merge and Julia factory corresponding two callbacks execute inside ten-repeat synchronized timer.100000+100000 positive sorted inputs, Float64 storage/comparison and integer path searches; no ordinary floating arithmetic denominator inferred. Resets/path/outputoracle and guard checks outside. Original standalone host paths uncalled.', 'mergeVS': 'Selected StudyFixture execute invokes original convert_vertical_slash_indexes host wrapper that launches mergehead kernel; Julia exact callback maps same metadata operator. Fourfixture/causal-mode paths have separate repeat timers summed. Int32/Int64 sparse attention metadata/index merge only, not actual attention/softmax arithmetic. Setup, reset, actualinput/guards/CPUoracle outside; sparse metadata interval semantics do not certify attention read validity.'}
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
  patches=[]
  for si,s in enumerate(r['stages']):
   if not s.get('source_callsite'):
    continue # vendor placeholders stay unresolved, never inferred from API name
   path=Path(s['source_callsite']['path']);legacy=any(x.endswith('-cuda')for x in path.parts)
   excluded=None
   if m=='cuda':
    if b=='matrixT':excluded='uninvoked_original_main_without_exact_function_pointer_candidate_join'
    elif b=='marchingCubes':excluded='untimed_scientific_preprocessing'if si<4 else 'uninvoked_legacy_main'if si>=5 else None
    elif b=='mask':excluded='uninvoked_legacy_main'if si>=6 else None
    elif b=='match':excluded='uninvoked_original_or_archived_standalone_main'if si>=10 else None
    elif b=='matern':excluded='untimed_warmup'if si==0 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='matrix-rotate':excluded='untimed_warmup'if si==0 else None
    elif b=='mcmd':excluded='uninvoked_archived_force_wrapper'if si==1 else None
    elif b=='md':excluded='untimed_warmup'if si==0 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='mdh':excluded='uninvoked_legacy_main'if si>=3 else None
    elif b=='meanshift':excluded='argv_unselected_base_branch'if si==1 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='medianfilter':excluded='untimed_warmup'if si==0 else 'uninvoked_legacy_main'if si>=2 else None
    elif b=='memtest':excluded='untimed_intentional_fault_audit'if si==3 else None
    elif b=='merge':excluded='uninvoked_legacy_main'if si>=2 else None
    elif legacy and b not in ['mergeVS']:excluded='uninvoked_legacy_main'
   else:
    if b=='marchingCubes'and si<4:excluded='untimed_scientific_preprocessing'
    if b=='mask'and m=='cuda_jl'and si==6:excluded='uninvoked_legacy_main'
    if b in ['maxpool3d','md','medianfilter']and m=='cuda_jl'and si>=1:excluded='uninvoked_legacy_main'
    if b=='matrixT'and m=='cuda_jl'and si>=1:excluded='uninvoked_legacy_host_wrapper'
    if b=='meanshift'and ((m=='ka'and si==0)or(m=='jacc'and si==1)):excluded='argv_unselected_base_branch'
   kind=excluded or 'timed_semantic_family';included=excluded is None
   updates={'semantic_family_in_primary_ROI':included,'ROI_membership_status':'source_timed_closure_inclusion_proved'if included else'source_untimed_audit_exclusion_proved'if excluded=='untimed_intentional_fault_audit'else'source_setup_exclusion_proved'}
   if not included:updates.update(model_kernel_regex=None,regex_status='disabled_'+excluded)
   if included and b=='merge'and m=='cuda_jl':
    name=['diagonal','window'][si];updates.update(model_kernel_regex=r'\b'+name+r'(?=[(<])',resolved_callable_candidates=[name],regex_status='exact_source_ordered_two_launch_factory')
   if included and b=='marchingCubes'and m=='cuda_jl':
    updates.update(model_kernel_regex=r'\bkernel5(?=[(<])',resolved_callable_candidates=['kernel5'],regex_status='exact_factory_ops5_generation_timed_only')
   if included and b=='memtest'and m=='jacc':
    updates.update(model_kernel_regex=r'_parallel_for_cuda[^\n]*\b(?:init_|move_|check_)(?=[,<(])',resolved_callable_candidates=['init!','move!','check!'],regex_status='exact_Source_Symbol_dispatch_all_three_callbacks')
   if included and b=='matrixT'and m in ['cuda_jl','jacc']:
    names=['copy_kernel!','copy_shared_kernel!','transpose_coarse_kernel!','transpose_fine_kernel!','transpose_naive_kernel!','transpose_coalesced_kernel!','transpose_pad_kernel!','transpose_swizzle_kernel!','transpose_diagonal_kernel!']if m=='cuda_jl'else['matrix_'+str(j)+'!'for j in range(9)]
    regex=(r'\b(?:'+ '|'.join(n.replace('!','_')for n in names)+r')(?=[(<])')if m=='cuda_jl'else(r'_parallel_for_cuda[^\n]*\bmatrix_[0-8]_(?=[,<(])')
    updates.update(model_kernel_regex=regex,resolved_callable_candidates=names,regex_status='exact_source_nine_entry_factory_tuple_kind_dispatch')
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
assert sha(legacy_path.read_bytes())==legacy_authority['compressed_sha256']
assert sha(legacy_receipt_path.read_bytes())==legacy_authority['receipt']['sha256']
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'legacy_authority_exclusion_check':legacy_authority,'schema_version':1,'scope':'Private batch10 source-only enrichment sixteen m-prefix named cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))

fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
