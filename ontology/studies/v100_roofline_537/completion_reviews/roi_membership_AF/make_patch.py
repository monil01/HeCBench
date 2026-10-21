import pathlib,json,gzip,hashlib,re,datetime
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=P(__file__).parent
NAMES=['accuracy','affine','adam','aobench','black-scholes','axpby'];parent=S/'roi_reviews/batch003_names1_80.json.gz';packed=parent.read_bytes();raw=gzip.decompress(packed);doc=json.loads(raw);parents={(r['benchmark'],r['model']):r for r in doc['records']}
def sha(b):return hashlib.sha256(b).hexdigest()
def binding(p):p=P(p);b=p.read_bytes();d={'path':str(p),'sha256':sha(b),'bytes':len(b)};return dict(d,uncompressed_sha256=sha(gzip.decompress(b)))if p.suffix=='.gz'else d
def load(p):return json.loads(gzip.decompress(P(p).read_bytes()))
def evidence(p,start,end,bindings):
 p=P(p).resolve();bs=p.read_bytes();h=sha(bs);matches=[(k,v)for k,v in bindings.items()if P(k).resolve()==p and v==h]
 if not matches:raise ValueError('Source absent/changed in selected frozen profile binding '+str(p))
 lines=bs.decode().splitlines();return {'path':str(p),'source_sha256':h,'profile_binding_paths':[k for k,v in matches],'start_line':start,'end_line':end,'exact_lines':'\n'.join(lines[start-1:end])}
def around(p,needle,b,back=2,forward=3):
 lines=P(p).read_text().splitlines();hits=[i for i,l in enumerate(lines)if needle in l]
 if not hits:raise ValueError('Missing evidence needle '+str(p)+' '+needle)
 i=hits[-1];return evidence(p,max(1,i+1-back),min(len(lines),i+1+forward),b)
# Hand reviewed callback symbols and explicit dispatch paths; never select families by
# observed profiler symbol alone. Matching below checks this source decision only.
FAMILY={'accuracy':('accuracy_kernel','accuracy_kernel_','accuracy_ka_','accuracy_jacc_'),'affine':('affine','affine_kernel_','affine_ka_','affine_jacc_'),'adam':('adam','adam_kernel_','adam_ka_','adam_jacc_'),'aobench':('render_kernel','render_kernel_','ao_kernel_','ao_kernel_'),'black-scholes':('getOutValOption','bs_kernel_','bs_ka_','bs_jacc_'),'axpby':('multi_tensor_apply_kernel','axpby_kernel_','axpby_ka_','axpby_jacc_')}
PREC={'accuracy':'FP32 input comparisons with Int32 reduction/atomics; no fabricated FP work.','affine':'UInt16 image storage; FP32 coordinate/interpolation expressions and integer indexing; source constants may fold SFU, no emitted-operation claim.','adam':'FP32 optimizer state and update; power/sqrt/division may include SFU/library lowering, not complete scalar-FLOP workload.','aobench':'FP32 geometry/shading and quantized UInt8 RGB; transcendental/SFU and control work not summarized by scalar-FLOP counts.','black-scholes':'FP32 pricing expressions including log/exp/sqrt/erf approximation; Float64 independent host oracle is outside timing, no FP64 device precision inferred.','axpby':'FP32 x+y / alpha=beta=1 multi-tensor update; indexing/control and memory traffic also present.'}
patches=[];aliases=[];unresolved=[];slots=[]
for n in NAMES:
 ledgerpath=S/'benchmarks'/f'{n}.json';ledger=json.loads(ledgerpath.read_text())
 for mi,m in enumerate(['cuda','cuda_jl','ka','jacc']):
  r=parents[n,m];state=ledger['models'][m];folder=(W/state['kernel_observations']['path']).parent;plan=load(folder/'plan.json.gz');prov=load(folder/'profile_provenance.json.gz');derived=load(folder/'derived.json.gz');bs=dict(plan['execution_bindings']);bs.update(prov.get('additional_current_profile_input_bindings',{}));stage=r['stages'][0];sym=FAMILY[n][mi]
  if state['profiling_status']!='profile_process_completed' or prov['status']!='profile_process_completed' or prov.get('binding_recheck_after')!='unchanged':raise ValueError('Unqualified selected profile '+n+'/'+m)
  if plan['configuration_id']!=r['configuration_id']or plan['reference_result_sha256']!=r['reference_result']['sha256']or plan['raw_config_sha256']!=r['raw_config_sha256']:raise ValueError('Contract mismatch')
  cr=S/'selected_contracts'/n
  if sha(gzip.decompress((cr/'result.json.gz').read_bytes()))!=plan['reference_result_sha256']or sha(gzip.decompress((cr/'config.json.gz').read_bytes()))!=plan['raw_config_sha256']:raise ValueError('Frozen contract bytes mismatch')
  if m=='cuda':regex=r'^(?:void )?'+sym+r'(?=[(<])'
  elif m=='ka':regex=r'^gpu_'+sym+r'\('
  elif m=='cuda_jl':regex=r'^'+sym+r'\('
  else:regex=r'^_parallel_for_cuda(?:_MN)?\([^\n]*\b'+sym+r'(?=[,<(])'
  if n=='axpby'and m=='cuda':regex=r'^void multi_tensor_apply_kernel<TensorListMetadata<\(int\)3>, AxpbyFunctor<float, float, float>, float, float, int>\('
  ids=[x['identity']for x in derived['records']];matches=[x for x in ids if re.search(regex,x['Kernel Name'])]
  if len(matches)!=1:raise ValueError('Cannot uniquely match intended observed family '+n+'/'+m+' '+regex)
  # Candidate aliases are source inventory artifacts, not a second operator. Neutralize
  # overlapping matching only, preserving source text/status and old hash in parent.
  effective={stage['stage_id']:regex}
  for other in r['stages'][1:]:
   pat=other.get('model_kernel_regex')
   if pat and re.search(pat.replace('\\\\','\\'),matches[0]['Kernel Name']):
    aliases.append({'benchmark':n,'model':m,'stage_id':other['stage_id'],'old_model_kernel_regex':pat,'set':{'model_kernel_regex':None,'kernel_family_alias_of':stage['stage_id']},'reason':'Duplicate source inventory callsite for same scientific family in an imported/renamed original main; preserve candidate evidence but do not let it create a second family match or claim a distinct measured phase.'});effective[other['stage_id']]=None
   else:effective[other['stage_id']]=pat
  for ident in matches:
   owners=[k for k,v in effective.items()if v and re.search(v,ident['Kernel Name'])]
   if owners!=[stage['stage_id']]:raise ValueError('Still ambiguous after proposed alias treatment')
  sources=[];root=W/'src'/(n+'-julia');chain=[]
  if m=='cuda':
   entry=P(next(x for x in plan['build_command']if x.endswith('.cu')));sources+=[evidence(entry,1,min(6,len(entry.read_text().splitlines())),bs),around(entry,'auto launch='if n!='aobench'else'auto run=',bs,0,3),around(entry,'auto start=',bs,1,2)]
   if n=='axpby':
    h=W/'src/axpby-cuda/multi_tensor_apply.cuh';sources += [around(h,'__global__ void multi_tensor_apply_kernel',bs,1,4),around(h,'multi_tensor_apply_kernel<<<',bs,2,3)];chain=['selected study_cuda.cu main -> launch lambda -> multi_tensor_apply<3>(...,AxpbyFunctor<float,float,float>(),1.f,1.f,-1) -> metadata batching -> multi_tensor_apply_kernel -> AxpbyFunctor','launch() runs inside the five-trial steady_clock interval and repeats loop, followed by cudaDeviceSynchronize before stop; full readback/check/write occurs after stop.']
   else:chain=['Selected native adapter renames imported original main; its own main constructs the named launch lambda and calls it from the five-trial synchronized steady_clock repeats interval.','Kernel callback symbol in the adapter identifies the family; the old imported-main launch is a source alias, not proof of an independently executed callsite.']
  else:
   entry=P(next(x for x in plan['profile_target_command']if x.endswith('.jl')));lns=entry.read_text().splitlines();sources += [evidence(entry,1,min(7,len(lns)),bs),evidence(entry,max(1,len(lns)-8),len(lns),bs)]
   common=root/'study_common.jl';cl=common.read_text().splitlines()
   if n=='aobench':sources += [evidence(common,1,min(9,len(cl)),bs)];chain=['Selected entry passes its named callback closure to ao_study(factory).','ao_study constructs run=factory(img), performs one untimed run, then executes run() R=100 times between start=time_ns() and synchronized elapsed calculation in each of five trials.','Reset/fill occurs before start; CPU readback/geometry-oracle/guards and file output occur after stop.']
   else:
    sources += [evidence(common,max(1,len(cl)-7),len(cl),bs),evidence(W/'src/_portable_study/OwnedStudy.jl',6,22,bs)]
    chain=['Selected entry registers the named callback through factory/launch function and invokes its specific study controller.','The study controller constructs launch!/launch using those exact input arrays/parameters and passes it to OwnedStudy.run_trials!.','OwnedStudy performs untimed reset+launch+synchronize, then reset+synchronize outside start; launch!() repeats inside start/time_ns..synchronize/elapsed for each of five trials; readback and numerical checks follow elapsed capture.']
   if m=='jacc':chain.insert(1,'JACC.parallel_for receives the explicit named '+(sym[:-1]+'!')+' callback shown in the source snippet; callback argument type in observed compiled wrapper corroborates this exact dispatch, not generic closure naming alone.')
   if m=='ka':chain.insert(1,'KA instantiates the named @kernel and returns/invokes that instance at the explicit ndrange shown; the gpu_ wrapper symbol is corroboration, not timing proof by itself.')
   if n=='accuracy':chain.append('accuracy_study wraps fill!(output,0);kernel!() together in launch!: that per-repeat reset belongs to the timed closure; its separate framework fill family is not mapped/proved here.')
   if n=='axpby':chain.append('launch() binds the exact selected nt/chunk/tensor-length arrays and calls the model launch callback; no inferred native batch count or other input configuration coverage.')
  proof={'kind':'explicit_selected_source_call_chain_to_enclosing_timed_closure','sources':sources,'chain':chain,'profile_binding_gate':'Every source snippet SHA equals a path in this selected frozen plan.execution_bindings or provenance.additional_current_profile_input_bindings; provenance binding_recheck_after=unchanged.','regex_family_correspondence':{'model_kernel_regex':regex,'matched_selected_kernel_identities':matches,'unique_family_match_after_explicit_alias_adjustments':True,'new_compute_or_case_coverage_claim':False},'sample_phase':'unknown; first sample may be warmup','sample_in_primary_timed_invocation':None,'whole_ROI_coverage':False,'causal_bottleneck':'unknown','precision_qualification':PREC[n]}
  patch={'benchmark':n,'model':m,'stage_id':stage['stage_id'],'configuration_id':r['configuration_id'],'reference_result_sha256':plan['reference_result_sha256'],'raw_config_sha256':plan['raw_config_sha256'],'parent_stage_sha256':sha(json.dumps(stage,sort_keys=True,separators=(',',':')).encode()),'set':{'model_kernel_regex':regex,'semantic_family_in_primary_ROI':True,'ROI_membership_status':'source_timed_closure_inclusion_proved','ROI_membership_proof':proof},'selected_profile_bindings':{'ledger':binding(ledgerpath),'manifest':binding(folder/'manifest.json'),'plan':binding(folder/'plan.json.gz'),'provenance':binding(folder/'profile_provenance.json.gz'),'derived':binding(folder/'derived.json.gz'),'frozen_config':binding(cr/'config.json.gz'),'frozen_result':binding(cr/'result.json.gz')}}
  patches.append(patch);slots.append({'benchmark':n,'model':m,'stage_id':stage['stage_id'],'source_bound':True,'sample_phase':'unknown'})
  for other in r['stages'][1:]:unresolved.append({'benchmark':n,'model':m,'stage_id':other['stage_id'],'reason':'No separate source family membership certification proposed; any alias matching adjustment only prevents duplicate authority. Imported original main/source inventory does not establish an observed invocation phase.'})
out={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Private minimal amendment proposal to existing candidate records, not independent/overlapping review authority. Parent must merge only against exact parent hashes and preserve history.','parent_review':{'path':str(parent),'compressed_sha256':sha(packed),'uncompressed_sha256':sha(raw)},'matching_rules_source':binding(W/'tools/v100_roofline_summarize.py'),'selected_benchmarks':NAMES,'stage_patches':patches,'non_authority_regex_alias_adjustments':aliases,'unresolved_candidate_stages':unresolved,'summary':{'benchmark_count':len(NAMES),'proved_model_stage_rows':len(patches),'regex_alias_adjustments':len(aliases),'source_setup_or_audit_exclusions_proposed':0,'GPU_runs':0,'build_runs':0,'tests':0},'phase_and_applicability_qualification':'Family belongs to timed launch closure in exact selected source; observed first sample phase remains unknown. No actual retained invocation, full ROI, aggregate benchmark roofline, causal diagnosis, emitted precision or cross-case extrapolation. SFU/integer/control work may be omitted by scalar FP projections.','shared_effort_not_translation':True}
(O/'patch.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'patch_sha256':sha((O/'patch.json').read_bytes()),**out['summary']},indent=2))
