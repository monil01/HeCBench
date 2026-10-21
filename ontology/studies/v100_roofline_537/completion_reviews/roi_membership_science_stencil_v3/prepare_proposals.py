import json,gzip,hashlib,re,datetime
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent
N={'bilateral','boxfilter','bm3d','bspline-vgh','bezier-surface'}
def sha(x):return hashlib.sha256(x).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def canon(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def evidence(p):
 b=p.read_bytes();t=b.decode();return {**file(p),'start_line':1,'end_line':t.count('\n')+1,'start_byte':0,'end_byte':len(b),'snippet':t}
why={
'bilateral':'Selected native launch(r) has explicit r==3/6/9 template dispatch; radii[3]={3,6,9} visits all cases in each checked trial and calls launch after start and before device sync/elapsed. Julia prepare returns radius closure explicitly invoking named CUDA/KA/JACC kernel; study_main passes each r in (3,6,9) through run_radius! after t0 and before sync/elapsed. Output reset and readback outside each timed radius. Identical closure also executes warmup. Native renamed main and CUDAjl stripped trailing main() are dormant.',
'boxfilter':'Selected native launch lambda invokes row_kernel then col_kernel; checked repeat invokes lambda between start and device sync/elapsed. Julia factory returns ordered row/column callback (KA named instances; JACC explicit row_jacc!/col_jacc!); box_study passes it to OwnedStudy.run_trials!, whose repeats call launch! between start=time_ns and synchronized elapsed. Temp/output resets precede timer; oracle and readback follow. Native included renamed original main is dormant.',
'bm3d':'Selected native run lambda invokes seven explicit run_* wrappers, each direct kernel dispatch: matching,gather,DCT,hard threshold,IDCT,aggregation,final. Checked timer encloses run repeated R=10 and completion; num/den zeroing is INSIDE run and timer, other reset outside. Julia factory returns same ordered seven-stage closure passed to bm_study; run! zeroes num/den then invokes closure within measured ten repeats. KA kernels are instantiated into named handles; JACC callbacks explicitly name each function. Forward/inverse transform use Boolean false/true but one compiled Bool family; source branches proved, sampled Bool value unproved. Oracle/guards/readback outside timer.',
'bspline-vgh':'Native launch(Case,walker) binds bspline and is called for each two coefficient fields/four walkers inside per-walker repeat timer; sync before elapsed. Julia prepare returns walker closure capturing named vgh kernel/KA instance/JACC callback, passed to study_main; each fixtures loop visits walker0:3 and invokes launch!(walker) after t0 before sync/elapsed. Initial coefficient/basis creation, CPU tensor oracle, warmups and output readback excluded. Hessian/gradient padding remains defined sentinel, not separate timed algorithm. Native renamed historical_main dormant.',
'bezier-surface':'Native study launch explicitly calls BezierGPU with cubic degree3/output512 and is repeated after start before synchronized elapsed; included renamed main is dormant. Julia study binds factory-returned CUDA/KA/JACC Bezier closure to OwnedStudy.run_trials!: launch! called after start=time_ns before synchronized elapsed. XYZ fill/reset and Float64 Bernstein/source CPU checks/readback outside timer. CUDAjl include_string strips exit(main(ARGS)) so historical legacy launch dormant. Warmup calls same closure; sampled invocation phase unproved.'}
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
  if m=='cuda':paths.add(base/('study_cuda.cu'if b in ['bilateral','bspline-vgh']else'study_native.cu'))
  else:
   paths.add(base/'study_common.jl')
   if b in ['boxfilter','bezier-surface']:paths.add(W/'src/_portable_study/OwnedStudy.jl')
   if b=='boxfilter'and m=='cuda_jl':paths.add(base/'boxfilter-julia-cuda.jl/study.jl')
   if b=='bm3d':paths.add(base/'study_device.jl')
   if b=='bspline-vgh':paths.add(base/'vgh_math.jl')
  ev=[evidence(p)for p in sorted(paths)];bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}
  for e in ev:
   assert bindings.get(str(Path(e['path']).resolve()))==e['sha256'],('selected source binding gap',b,m,e['path'])
   sources[e['path']]=file(Path(e['path']))
  patches=[]
  for si,s in enumerate(r['stages']):
   dormant=(b=='bilateral'and m=='cuda'and si>=3)or(b=='bilateral'and m=='cuda_jl'and si==1)or(b=='boxfilter'and m=='cuda'and si>=2)or(b=='bspline-vgh'and m=='cuda'and si==1)or(b=='bezier-surface'and m in ['cuda','cuda_jl']and si==1)
   updates={'semantic_family_in_primary_ROI':not dormant,'ROI_membership_status':'source_setup_exclusion_proved'if dormant else'source_timed_closure_inclusion_proved'}
   if dormant:updates.update(model_kernel_regex=None,regex_status='disabled_dormant_legacy_entrypoint_not_selected_ROI')
   if b=='bilateral'and m=='cuda'and si<3:
    radius=[3,6,9][si];updates.update(model_kernel_regex=r'\bbilateralFilter<(?:\(int\))?'+str(radius)+r'>(?=[(<])',regex_status='source_template_specialization_alias_disambiguated_not_phase_proof')
   if b=='bm3d'and m in ['cuda_jl','jacc']and si==4:
    updates.update(model_kernel_regex=None,regex_status='alias_same_transform_Bool_family_as_source_callsite_003_phase_not_resolved',alias_of_stage_id=r['stages'][2]['stage_id'])
   if b=='bezier-surface'and m=='cuda_jl'and si==0:
    updates.update(model_kernel_regex=r'(?:\bbezier_gpu_kernel(?=[(<])|_Z17bezier_gpu_kernel(?=\d))',regex_status='source_symbol_and_selected_mangled_symbol_alias_not_invocation_proof')
   updates['ROI_membership_proof']={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':'dormant_legacy_entrypoint'if dormant else'timed_semantic_family','exact_stage_callsite':s['source_callsite'],'reason':why[b],'evidence':ev,'timed_closure_boundaries':[{'path':e['path'],'sha256':e['sha256'],'timer_lines':[{'line':i+1,'text':line}for i,line in enumerate(e['snippet'].splitlines())if any(v in line for v in ['time_ns()', 'steady_clock::now()', 'run_trials!(', 'bm_study(factory)', 'exit(study_main(', 'bezier_study(', 'box_study(factory)'])]}for e in ev],'selected_result':r['reference_result'],'selected_config':r['raw_config'],'actual_profile_source_bindings':file(pp),'selected_profile_manifest':file(mp),'selected_counter_derivation':file(op),'profile_target_command':plan['profile_target_command'],'precision_scope':'Source-declared FP32 arithmetic plus integer addressing/storage and special functions where present; source precision does not certify actual SASS instruction class.','limitations':['Source inclusion only: first captured invocation may be warmup/setup; sample phase remains unproved.','No complete elapsed-ROI or complete launch census, multiplicity extrapolation, numerical recertification or causal bottleneck claim.','Alias nulling avoids repeated same-family candidates; it does not remove source calls or hide conditional branches.']}
   patches.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canon(s),'updates':updates})
  revised=[dict(s,**patches[i]['updates'])for i,s in enumerate(r['stages'])];matches=[]
  for ob in obs:
   name=ob['identity']['Kernel Name'];hits=[s['stage_id']for s in revised if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],name)]
   assert len(hits)<=1,(b,m,name,hits)
   matches.append({'identity':ob['identity'],'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  parents[str(parent)]={'compressed_sha256':sha(pb),'uncompressed_sha256':sha(raw)}
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canon(r),'stage_patches':patches,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'ledger':{'path':str(lp),'sha256':sha(lb)},'manifest':file(mp),'plan':file(pp),'derivation':file(op)})
# Before/after bindings establish no selected inputs changed during the private read-only review.
for path,e in sources.items():assert sha(Path(path).read_bytes())==e['sha256']
for path,e in parents.items():assert sha(Path(path).read_bytes())==e['compressed_sha256']
for r in selected:
 for key in ['ledger','manifest','plan','derivation']:assert sha(Path(r[key]['path']).read_bytes())==r[key]['sha256']
out={'schema_version':1,'scope':'Private source-only enrichment five named scientific cases; root merge required','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':withheld,'merging_policy':'Recheck parent compressed/raw SHA and canonical record/stage SHA and selected/source bindings; preserve all historical gzip bytes. No overlapping review authority.','before_after_bindings_stable':True,'GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
p=O/'proposed_stage_patches.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'models':len(proposals),'stages':sum(len(r['stage_patches'])for r in proposals),'sources':len(sources),'sha256':sha(p.read_bytes())}))
