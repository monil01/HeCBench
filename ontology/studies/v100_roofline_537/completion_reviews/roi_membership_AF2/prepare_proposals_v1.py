import pathlib,json,gzip,hashlib,re,datetime,copy
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=P(__file__).parent
N={'atan2','bincount','bitpacking','adamw','bonds','feynman-kac'}
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def file(p):p=P(p);b=p.read_bytes();r={'path':str(p),'sha256':sha(b)};return dict(r,uncompressed_sha256=sha(gzip.decompress(b)))if p.suffix=='.gz'else r
def frozen(p):return json.loads(gzip.decompress(P(p).read_bytes()))
parents={};records={}
for p in sorted((S/'roi_reviews').glob('*.json.gz')):
 packed=p.read_bytes();raw=gzip.decompress(packed);d=json.loads(raw)
 for ri,r in enumerate(d['records']):
  if r['benchmark']in N:
   parents[str(p)]={'compressed_sha256':sha(packed),'uncompressed_sha256':sha(raw)};records[r['benchmark'],r['model']]=(p,ri,r)
family_info={
'atan2':{'parameters':'Selected argv n4096/repeat10. main loops case=1:3 in each of five trials; each maps explicitly to f/i/s approximation callback and distinct output array; launch repeats inside synchronized timer.','scientific_family':'Polynomial/rational atan2 approximations for float/int/short outputs.','precision':'Float32 input and approximation arithmetic with integer conversion/control; selected f degree15 helper includes source Float64 promotion in Julia. Float64 artifact widening is not proof of all-FP64 computation; emitted scalar mix requires counters. Do not infer SFU atan instruction merely from benchmark name.'},
'bincount':{'parameters':'All selected profile_target_command arrays end in shared, and controller/factory predicates test that literal. n262144, bins768, repeats10. Only SHARED branch is executed under selected contract; GLOBAL candidate remains unproved/unselected, not called setup/audit.','scientific_family':'Shared-memory histogram with local/global Int32 atomic counts.','precision':'Float32 bin arithmetic/comparison and integer atomic/reduction/control; count storage is Int32. No fabricated FP-equivalent work for atomics.'},
'bitpacking':{'parameters':'Fixed six sizes2,123,3411,83621,872163,1049603; each timed run calls scan then finalize then pack. Controller resets packed output before timer and runs full configuration+packing1000times per size. Source sequence membership does not establish actual sampled launch ordinals or full trace.','scientific_family':'Integer min/max configuration reduction and buffered variable-width bitstream packing.','precision':'Int32/UInt32/UInt64 arithmetic, shifts, reduction/control/shared memory; scalar FP roof not applicable as arithmetic work model.'},
'adamw':{'parameters':'n4096 packed pairs,64groups×64threads. run! iterates the two fixed PARAMS tuples in order. study resets original six arrays before start, runs the full two-step callback, synchronizes then divides elapsed by2. Packed state and evolving second-step data are preserved.','scientific_family':'Two-step quantized AdamW optimizer with block scale reduction and nibble re-encoding.','precision':'Source FP32 parameter/update/scales with packed four-bit state in Int8/UInt8 storage, integer quantization/control and sqrt/division. Four-bit storage does not establish FP4 arithmetic or tensor mode.'},
'bonds':{'parameters':'Exactly four source CASES n1,17,257,1000000. Each state reset and pre-capture precedes timer; one named bonds launch plus completion is timed per case. Post-capture/oracle follows stop. Three observed geometries do not identify all four case invocations or prove complete ROI.','scientific_family':'Dated financial bond cashflow/price/yield calculation and safeguarded yield search.','precision':'Source FP32 typed financial inputs/results with integer dates/branches and pow/exp-like operations. Float64 independent oracle/serialized captures are not a device FP64 claim; scalar FP counters omit SFU/control effects.'},
'feynman-kac':{'parameters':'NI17,NJ9, paths32, seed123456789, a2,b1,h.001,rthsqrt(2h). Each five-trial record averages3 separately timed full stochastic kernels. Count/error resets synchronize before timer; readback/oracle follows elapsed. Fixed per-point RNG/firstsamplephase remain distinct.','scientific_family':'Feynman-Kac stochastic random walk with FP64 error and integer count atomics.','precision':'FP64 stochastic potential/weight/error arithmetic with Int32 RNG/control/count and exp/pow effects. Scalar FP projection omits integer/SFU and atomics cost; no causal bottleneck inferred.'}}
proposals=[];sources={};selected=[];unresolved=[]
for n in sorted(N):
 for m in ['cuda','cuda_jl','ka','jacc']:
  parent,ri,r=records[n,m]
  if any(x.get('ROI_membership_status')=='source_timed_closure_inclusion_proved'for x in r['stages']):raise ValueError('Already proved family '+n+'/'+m)
  ledgerpath=S/'benchmarks'/f'{n}.json';state=json.loads(ledgerpath.read_text())['models'][m];folder=(W/state['kernel_observations']['path']).parent;plan=frozen(folder/'plan.json.gz');prov=frozen(folder/'profile_provenance.json.gz');observed=frozen(folder/'derived.json.gz')['records'];bound={P(k).resolve():(v,'plan.execution_bindings')for k,v in plan['execution_bindings'].items()}
  for k,v in prov.get('additional_current_profile_input_bindings',{}).items():bound.setdefault(P(k).resolve(),(v,'provenance.additional_current_profile_input_bindings'))
  if state['profiling_status']!='profile_process_completed'or prov['status']!='profile_process_completed'or prov.get('binding_recheck_after')!='unchanged':raise ValueError('Selected successful/hash-rechecked profile required')
  if (plan['configuration_id'],plan['reference_result_sha256'],plan['raw_config_sha256'])!=(r['configuration_id'],r['reference_result']['sha256'],r['raw_config_sha256']):raise ValueError('Selected contract mismatch')
  for key in ['reference_result','raw_config']:
   if sha(P(r[key]['path']).read_bytes())!=r[key]['sha256']:raise ValueError('Contract bytes drift')
  evidence=[]
  def span(path,start=1,end=None):
   p=P(path).resolve();bs=p.read_bytes();h=sha(bs)
   if p not in bound or bound[p][0]!=h:raise ValueError('Unbound selected source '+str(p))
   lines=bs.decode().splitlines(keepends=True);end=len(lines)if end is None else min(end,len(lines));text=''.join(lines[start-1:end]);v={**file(p),'source_binding_origin':bound[p][1],'start_line':start,'end_line':end,'start_byte':len(''.join(lines[:start-1]).encode()),'end_byte':len(''.join(lines[:end]).encode()),'snippet':text};sources[str(p)]=file(p);evidence.append(v);return v
  def tail(path,needle,back=1):
   lines=P(path).read_text().splitlines();hits=[i for i,l in enumerate(lines)if needle in l]
   if not hits:raise ValueError('Missing exact timer/controller '+str(path)+' '+needle)
   return span(path,max(1,hits[-1]+1-back))
  base=W/'src'/(n+'-julia')
  if m=='cuda':
   entry=P(next(x for x in plan['build_command']if x.endswith('.cu')));span(entry,1,min(5,len(entry.read_text().splitlines())))
   if n=='atan2':tail(entry,'int main(',0);why='Selected adapter main defines launch(c) with explicit c1/2/3 callbacks; each case callback is called inside steady_clock repeats then synchronized elapsed; readback follows stop. Imported hecbench_original_main is not invoked.'
   elif n=='bincount':tail(entry,'const bool shared=',0);span(W/'src/bincount-cuda/main.cu',21,23);why='Selected argv shared makes native shared=true. launch lambda chooses DeviceMemoryType::SHARED (enum0); lambda runs in five-trial steady_clock repeats. Pretrial reset before start, output readback/check after elapsed. GLOBAL branch is not generalized from shared proof.'
   elif n=='bitpacking':tail(entry,'auto launch=',0);span(W/'src/bitpacking-cuda/kernels.cu',282,312);span(W/'src/bitpacking-cuda/kernels.cu',324,342);span(W/'src/bitpacking-cuda/kernels.cu',353,408);why='Selected main launch(c) calls bitPackInternal<int,uint32_t,int>. That host function calls bitPackConfigLaunch (scan then finalize) and bitPackFixedBitAndMinInternal -> bitPackLaunch (pack). Full launch(c) repeats between steady_clock start/synchronized elapsed for every fixed size; all three families belong to that interval.'
   elif n=='adamw':tail(entry,'int main(',0);why='Selected adapter main constructs run lambda iterating step0/1 over fixed parameter bit patterns, calls fused_4bit_kernel<float,64>. Trial reset is before timer; entire run+completion inside timer, elapsed/2 then full state capture/oracle. Renamed legacy_main is not invoked.'
   elif n=='bonds':tail(entry,'struct State',0);why='State::run launches bonds(a,r,c.n) and synchronizes. In five-trial state loop, reset+capture_pre precede start; s->run is enclosed by steady_clock now/elapsed; check_post follows. Four selected states bind exact case set, no actual sample-to-case assignment.'
   else:tail(entry,'int main()',0);why='Selected main defines run with fk selected constants. For each five-trial record, reset before steady_clock then run+completion inside timer, divided by3 across three repetitions. Readback/check follows. Renamed legacy_main never called.'
  else:
   entry=P(next(x for x in plan['profile_target_command']if x.endswith('.jl')))
   if n=='atan2':tail(entry,'function launch(',0);why='Entry main executes all cases1:3 and the explicit launch(case) dispatches f/i/s named callbacks. Each callback is repeated inside start=time_ns/synchronized elapsed, then readback. Parameters bind selected4096/10 argv. No inference from profiler name alone.'
   elif n=='bincount':span(entry);tail(base/'study_common.jl','function bincount_study(',0);span(W/'src/_portable_study/OwnedStudy.jl',6,22);why='Entry factory chooses shared callback from selected shared argv. bincount_study constructs launch! then OwnedStudy runs that exact closure in five-trial timed repeat loops; output reset is outside start. Explicit JACC callback/KA shared instance is source-bound; GLOBAL remains unselected.'
   elif n=='bitpacking':span(entry);tail(base/'study_common.jl','function study(',0);why='Entry passes run! to study. run! dispatches explicit scan!,finalize!,pack! in order. study loops fixed six sizes, resets before timer, executes whole run!1000times inside timer and synchronizes before elapsed. Callback names/KA instances and source arguments identify each family independent of generic wrapper name.'
   elif n=='adamw':tail(entry,'function run!',0);span(entry,1,6);span(base/'study_common.jl',1,6);tail(base/'study_common.jl','function study(',0);why='Entry run! loops two PARAMS tuples and calls named kernel! explicitly (CUDA/KA/JACC). Shared study constructs initial arrays/template, warms once, resets before timer; run! plus synchronization is timed and elapsed divided by2. Packed-state update is sequential, no firstsample step identity asserted.'
   elif n=='bonds':span(entry);span(base/'study_cases.jl');tail(base/'study_common.jl','for s in states',0);why='Selected entry passes exact allocation and launch(dev,n) callbacks to study. Study iterates fixed CASES, resets/captures before timer then launch+completion inside start/elapsed; post-check after stop. JACC callback bond_kernel and KA bond_kernel instance are explicit. No inference from generic closure or FP64 artifact storage.'
   else:span(entry);tail(base/'study_common.jl','const NI=',0);why='Entry run! dispatches named kernel! at explicit17x9 domain and passes to study. Shared controller constructs fixed seed/paths/FP64 parameters, warms once, then resets/synchronizes before each timer; run!+completion timed three times. Count/error readback and independent CPU point oracle follow elapsed. gpu_fill reset family is not annotated here.'
  # Hand-reviewed owner stages and exact callable identities. All other source
  # candidates remain unresolved unless a precise same-family alias is neutralized.
  if n=='atan2':owners=[(i,('compute_'+c if m=='cuda'else'compute_'+c+'_kernel_'))for i,c in enumerate('fis')]
  elif n=='bincount':owners=[(1 if m=='ka'else 0,'bincount'if m=='cuda'else'bincount_shared_ka_'if m=='ka'else'bincount_shared_jacc_'if m=='jacc'else'bincount_shared_kernel_')]
  elif n=='bitpacking':owners=[(i,(['bitPackConfigScanKernel','bitPackConfigFinalizeKernel','bitPackKernel'][i]if m=='cuda'else['scan_','finalize_','pack_'][i]))for i in range(3)]
  elif n=='adamw':owners=[(0,'fused_4bit_kernel'if m=='cuda'else'kernel_')]
  elif n=='bonds':owners=[(0,'bonds'if m=='cuda'else'bond_kernel')]
  else:owners=[(0,'fk'if m=='cuda'else'kernel_')]
  stagepatch=[];revised=copy.deepcopy(r['stages']);ids=[x['identity']for x in observed]
  for si,sym in owners:
   st=r['stages'][si]
   if m=='cuda':regex=r'^(?:void )?'+sym+r'(?=[(<])'
   elif m=='cuda_jl':regex=r'^'+sym+r'\('
   elif m=='ka':regex=r'^gpu_'+sym+r'\('
   else:regex=r'^_parallel_for_cuda(?:_MN)?\([^\n]*\b'+sym+r'(?=[,<(])'
   if n=='bincount'and m=='cuda':regex=r'^void bincount<int, float, int, \(DeviceMemoryType\)0>\('
   if n=='adamw'and m=='cuda':regex=r'^void fused_4bit_kernel<float, \(int\)64>\('
   matches=[x for x in ids if re.search(regex,x['Kernel Name'])]
   if not matches:raise ValueError('No intended selected family '+n+'/'+m+'/'+sym)
   proof={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':'timed_scientific_family','exact_stage_callsite':st['source_callsite'],'reason':why,'input_parameter_state_qualification':family_info[n]['parameters'],'scientific_family':family_info[n]['scientific_family'],'precision_and_non_scalar_work_qualification':family_info[n]['precision'],'evidence':evidence,'actual_profile_source_bindings':file(folder/'plan.json.gz'),'additional_profile_source_bindings':file(folder/'profile_provenance.json.gz'),'selected_result':r['reference_result'],'selected_config':r['raw_config'],'selected_profile_manifest':file(folder/'manifest.json'),'selected_counter_derivation':file(folder/'derived.json.gz'),'profile_target_command':plan['profile_target_command'],'regex_family_correspondence':{'regex':regex,'matched_observed_identities':matches,'meaning':'Unique semantic family owner; several observed geometry rows may belong to the same family. Not a complete launch/case census.'},'sample_phase':'unknown','sample_in_primary_timed_invocation':None,'retained_trial':None,'limitations':['Same family executes warmup and timed calls; actual observed first sample phase is unproved.','Source inclusion does not prove whole elapsedROI, ordered complete launch census, full set of cases captured, phase aggregate or causal bottleneck.','No GPU execution, SASS lowering or tensor precision certification; source precision qualifiers do not replace counter modes.','No input/configuration or other-family extrapolation.']}
   up={'model_kernel_regex':regex,'semantic_family_in_primary_ROI':True,'ROI_membership_status':'source_timed_closure_inclusion_proved','ROI_membership_proof':proof};stagepatch.append({'stage_index':si,'stage_id':st['stage_id'],'expected_original_stage_canonical_sha256':canonical(st),'updates':up});revised[si].update(up)
   for oi,other in enumerate(r['stages']):
    if oi==si or oi in [i for i,_ in owners]:continue
    # GLOBAL shared=false is a distinct unselected family, never an alias.
    if n=='bincount'and any('GLOBAL'in c or'global'in c for c in other.get('callable_candidates',[])):continue
    oldpat=other.get('model_kernel_regex')
    same_symbol=other.get('callable_candidates')==st.get('callable_candidates')
    if oldpat and same_symbol and any(re.search(oldpat.replace('\\\\','\\'),x['Kernel Name'])for x in matches):
     upa={'model_kernel_regex':None,'kernel_family_alias_of':st['stage_id']};stagepatch.append({'stage_index':oi,'stage_id':other['stage_id'],'expected_original_stage_canonical_sha256':canonical(other),'alias_only':True,'updates':upa,'alias_proof':{'reason':'Same exact callable candidate identity and observed family as proved owner; imported legacy/original main is renamed or not invoked by selected controller. Preserve source candidate evidence/status; disable duplicate name ownership only.','owner_stage_id':st['stage_id'],'callable_candidates':st['callable_candidates'],'evidence':evidence}});revised[oi].update(upa)
  mappings=[]
  for ident in ids:
   hits=[st['stage_id']for st in revised if st.get('model_kernel_regex')and re.search(st['model_kernel_regex'],ident['Kernel Name'])]
   if len(hits)>1:raise ValueError('Ambiguous proposed family ownership '+n+'/'+m+' '+str(hits))
   mappings.append({'identity':ident,'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  patched={x['stage_index']for x in stagepatch}
  for i,st in enumerate(r['stages']):
   if i not in patched:unresolved.append({'benchmark':n,'model':m,'stage_id':st['stage_id'],'reason':'Distinct unselected GLOBAL branch under shared-only command; no setup/audit false status or alternate-case inclusion inferred.'if n=='bincount'else'No separate family proof proposed; existing stage unchanged.'})
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':n,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canonical(r),'stage_patches':stagepatch,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']if k in r},'selected_observation_matching':mappings})
  selected.append({'benchmark':n,'model':m,'ledger_snapshot':file(ledgerpath),'manifest':file(folder/'manifest.json'),'plan':file(folder/'plan.json.gz'),'provenance':file(folder/'profile_provenance.json.gz'),'derivation':file(folder/'derived.json.gz'),'reference_result_sha256':plan['reference_result_sha256'],'raw_config_sha256':plan['raw_config_sha256'],'source_binding_recheck_after':prov['binding_recheck_after']})
# Refuse output on concurrently changed parent review bytes.
for p,h in parents.items():
 if file(p)['sha256']!=h['compressed_sha256']:raise ValueError('Parent changed during proof review; rebase required')
out={'schema_version':1,'scope':'A-F second private source-membership amendment proposal, six new names/all24models; no shared mutation or overlapping review authority.','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'unresolved_cases':unresolved,'merging_policy':'Verify exact parent compressed/raw SHA, canonical record/stage SHA and all selected capture/source bindings before merging into existing authority; preserve old review bytes/history. alias_only patches alter only regex/alias fields, no semantic membership/status. Source-proof rows require explicit timed closure evidence; never infer sample phase from family membership.','GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False,'shared_effort_not_translation':True}
(O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'records':len(proposals),'proved_inclusions':sum(not x.get('alias_only')for r in proposals for x in r['stage_patches']),'aliases':sum(bool(x.get('alias_only'))for r in proposals for x in r['stage_patches']),'sources':len(sources),'sha256':sha((O/'proposed_stage_patches.json').read_bytes())},indent=2))
