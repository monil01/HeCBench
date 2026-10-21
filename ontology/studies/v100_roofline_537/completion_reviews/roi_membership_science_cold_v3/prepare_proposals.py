"""Private source ROI proposal from short locked immutable metadata snapshots."""
import pathlib,json,gzip,hashlib,re,datetime,copy
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=P(__file__).parent;N={'axhelm','bicgstab','ccsd-trpdrv','clenergy','chemv'};snap=json.loads((O/'inputs_before/snapshot.json').read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def file(p):
 p=P(p);b=p.read_bytes();d={'path':str(p),'sha256':sha(b)}
 if p.suffix=='.gz':d['uncompressed_sha256']=sha(gzip.decompress(b))
 return d
def frozen(p):return json.loads(gzip.decompress(P(snap['files'][str(p)]['snapshot']).read_bytes()))
parents={};records={};sources={};patches=[];selected=[];unresolved=[]
for name in('batch003_names1_80.json.gz','batch003_names81_160_review.json.gz'):
 parent=S/'roi_reviews'/name;data=frozen(parent);ref=snap['files'][str(parent)];parents[str(parent)]={'compressed_sha256':ref['sha256'],'uncompressed_sha256':ref['uncompressed_sha256']}
 for ri,r in enumerate(data['records']):
  if r['benchmark']in N:records[r['benchmark'],r['model']]=(parent,ri,r)
INFO={
'axhelm':{'why':'Selected nativeonecase(fields) and Julia studycase invoke scalar forfields1 andfused3 forfields3. Both fieldscases are enumerated in warmup andeverytrial. start precedes allocation/H2D,100launches thenD2H/free andcompletion precede elapsed/100. Thus kernelmembership is inclusion in a broader allocation/transfer/free ROI, not an elapsedkernel-only aggregate.','params':'7elements,8×8threads,512points/element; fields(1,3),100repeats. JACC variablef explicitly selects scalar!/fused3! from fields; KA explicitinstance follows samepredicate. Both branches selected; no otherfield inferred.','precision':'FP32 source dfloat=float native and Float32arrays/0f0localaccumulators/typedsharedmemory and arithmetic Julia helm_body!, integerindexcontrol/workgroupbarriers. Neither FP64CPUerrorchecks nor outputstorage is the precisionproof.','family':'Scalar and three-field fused tensor-product Helmholtz operator'},
'ccsd-trpdrv':{'why':'Selected nativewrapper launch lambda and Julia study_launch! bind ccsd callback. Sharedcontroller enumerates3fixtures×18occupiedtriples inside repeat loop between start andcompletionelapsed. Outputzero/reset precedes start; divide elapsed byrepeat thenreadback/check follows. Original importeddriver main/hostfunction is not selected; same callback candidates are aliases only.','params':'nv33,no3,18triples,3fixedfixtures;54kernelinvocations/repeat sourcecount, selectedrepeatargv retained. KA guards b,c<=nv paddedgrid; JACC ndrange(nv,nv) passes explicit ccsd_jacc!+CCSD_SPEC. Sourcebranch for i!=k is CPUenergyaggregation afterstop, not guessed GPUexclusion.','precision':'Native double arithmetic, Julia Float64selectedarrays and ccsd_terms -1.0denominator/+/* terms, FP64energyoutputatomics; BigFloatCPUoracle is outside timer. Emittedscalar/atomiclowering unmeasured, no tensorcertification.','family':'CCSD triples-driver energy contributions'},
'clenergy':{'why':'Native timed cenergy call atselectedwrapperline38 is enclosed by perfragmenttimer, completion thenaccumulated seconds. Julia clenergy_study passes explicit launch callback, synchronizes/start/launch/completion/elapsed for everychunk. Nativewarmup cenergy atline28 shares family; disable duplicate owner viaalias only, never label samefamily untimed from firstcallsite.','params':'Grid64×8; atomcounts1,2,127,4001,64; chunksMAXATOMS4000 give6fragments across5fixtures. Freshoutputreset perfixture before fragmenttimers; accumulatedoutputacross2chunks for4001 preserved; inputpreparation/readback/guards/physicaloracle outside.','precision':'Source FP32coordinates/charge/8accumulators. Native rsqrtf versus Julia inv(sqrt(Float32expression)); SFU/divisioncost omitted from scalar add/mul/FMAroofline. Float64independentphysicaloracle outsidetimer, no emittedinstruction/SFUmode inferred.','family':'Coulomb electrostatic energy grid'},
'chemv':{'why':'Selectednative launchlambda calls chemv_kernel0 then1 in order, invoked inside timedrepeatloop. Julia factory returns closure with correspondingtwoexplicitcallbacks; chemv_study passes closure to OwnedStudy.run_trials!, reset beforestart, closure+completiontimed, readback/CPUcheckafter. CUDAjlstudy.jl include_string strips exit(main()) from importedCLI, so main.jl launchcandidates are same-family aliases not independenttimedfamilies.','params':'n370,oneinternalrepeat,12blocks×32threads/384logicalworkitems, exactFloat32alpha/beta literals. Bothstage0lower/diagonal+beta andstage1conjugateupper contributions execute eachclosure in order; branchbounds are sourceactual and no sample-to-row/subcase inference.','precision':'ComplexFloat components explicitly Float32 and alpha/beta arguments Float32, scalarreal/imagadd/mulproducts and Int32indices. ComplexF64CPUreference construction does not make GPUFP64; emittedFPmix remains counterevidence.','family':'Complex Hermitian matrix-vector product'}}
for n in sorted(N):
 for m in('cuda','cuda_jl','ka','jacc'):
  parent,ri,r=records[n,m];sel=snap['selected'][n+'/'+m];folder=P(sel['folder']);plan=frozen(folder/'plan.json.gz');prov=frozen(folder/'profile_provenance.json.gz');observed=frozen(folder/'derived.json.gz')['records'];ids=[x['identity']for x in observed]
  if prov['status']!='profile_process_completed'or prov.get('binding_recheck_after')!='unchanged':raise ValueError('Completedselectedboundcapture required')
  if(plan['configuration_id'],plan['reference_result_sha256'],plan['raw_config_sha256'])!=(r['configuration_id'],r['reference_result']['sha256'],r['raw_config_sha256']):raise ValueError('Selectedcontract mismatch '+n+'/'+m)
  for key in('reference_result','raw_config'):
   if file(r[key]['path'])['sha256']!=r[key]['sha256']:raise ValueError('Selectedcontract changed')
  bound={P(k).resolve():(v,'plan.execution_bindings')for k,v in plan['execution_bindings'].items()}
  for k,v in prov.get('additional_current_profile_input_bindings',{}).items():bound.setdefault(P(k).resolve(),(v,'provenance.additional_current_profile_input_bindings'))
  evidence=[]
  def span(p):
   p=P(p).resolve();raw=p.read_bytes();h=sha(raw)
   if p not in bound or bound[p][0]!=h:raise ValueError('Unboundselectedsource '+str(p))
   text=raw.decode();e={**file(p),'source_binding_origin':bound[p][1],'start_line':1,'end_line':len(text.splitlines()),'start_byte':0,'end_byte':len(raw),'snippet':text};evidence.append(e);sources[str(p)]=file(p);dest=O/'source_snapshots'/p.relative_to(W);dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.write_bytes(raw)
   return e
  base=W/'src'/(n+'-julia');selected.append({'benchmark':n,'model':m,'ledger_snapshot':snap['files'][sel['ledger']],'manifest':snap['files'][str(folder/'manifest.json')],'plan':snap['files'][str(folder/'plan.json.gz')],'provenance':snap['files'][str(folder/'profile_provenance.json.gz')],'derivation':snap['files'][str(folder/'derived.json.gz')],'selected_kernel_observations':sel['kernel_observations'],'reference_result_sha256':plan['reference_result_sha256'],'raw_config_sha256':plan['raw_config_sha256'],'source_binding_recheck_after':prov['binding_recheck_after']})
  if n=='bicgstab':
   span(base/'study_bridge.cu');span(base/('study_native.py'if m=='cuda'else'study_common.jl'))
   if m!='cuda':span(next(x for x in plan['profile_target_command']if x.endswith('.jl')))
   unresolved.append({'benchmark':n,'model':m,'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':n,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canonical(r),'stage_id':r['stages'][0]['stage_id'],'expected_original_stage_canonical_sha256':canonical(r['stages'][0]),'proposed_updates':None,'reason':'Existingvendor_operation aggregate has regexNone and is not an individuallysource-resolved kernelowner. Actualcusparse/cublas/CUBkernelidentities spananalysis,integerhelpers,reductions,arithmetic; cannot invent exactAPI→kernel/phase association fromnames. No existingstagechanged.','source_timed_API_context_only':{'reason':'Native study_native_timed encloses study_native recurrence; Julia timer encloses translated_solver including study_prepare+library primitives+finish. Resets/capture/check outside. FP64explicit cublasDdot/Dnrm2/Daxpy/Dscal and CUDA_R_64FSpMV/SpSV, plusintegeranalysis andhostconvergence. Actualbranchiterations and everykerneltoAPIassociation notproved here.','evidence':evidence,'sample_phase':'unknown','sample_in_primary_timed_invocation':None}});continue
  if m=='cuda':
   entry=P(next(x for x in plan['build_command']if x.endswith('.cu')));span(entry)
   if n=='axhelm':span(W/'src/axhelm-cuda/axhelmKernel.cpp')
   elif n=='clenergy':span(W/'src/clenergy-cuda/clenergy.cu')
   elif n=='chemv':span(W/'src/chemv-cuda/main.cu')
   else:span(W/'src/ccsd-trpdrv-cuda/ccsd_tengy.cu')
  else:
   entry=P(next(x for x in plan['profile_target_command']if x.endswith('.jl')));span(entry);span(base/'study_common.jl')
   if n=='axhelm':span(base/'study_kernel_bodies.jl')
   if n=='ccsd-trpdrv':span(base/'ccsd_math.jl')
   if n=='chemv':span(W/'src/_portable_study/OwnedStudy.jl')
   if entry.name=='study.jl':span(entry.with_name('main.jl'))
  if n=='axhelm':owners=[(0,'axhelm'),(1,'axhelm_n3')]if m=='cuda'else[(0,'scalar_'),(1,'fused3_')]if m!='jacc'else[(0,'(?:scalar_|fused3_)')]
  elif n=='ccsd-trpdrv':owners=[(0,'ccsd_kernel'if m=='cuda'else'ccsd_kernel_'if m=='cuda_jl'else'ccsd_ka_'if m=='ka'else'ccsd_jacc_')]
  elif n=='clenergy':owners=[(1 if m=='cuda'else 0,'cenergy'if m=='cuda'else'clenergy_cuda_'if m=='cuda_jl'else'clenergy_ka_'if m=='ka'else'clenergy_jacc_')]
  else:owners=[(i,'chemv_kernel'+str(i)+(''if m=='cuda'else'_'))for i in range(2)]
  revised=copy.deepcopy(r['stages']);stagepatch=[]
  for si,sym in owners:
   st=r['stages'][si]
   regex=(r'^(?:void )?'+sym+r'(?=[(<])')if m=='cuda'else(r'^'+sym+r'\(')if m=='cuda_jl'else(r'^gpu_'+sym+r'\(')if m=='ka'else(r'^_parallel_for_cuda(?:_MN)?\([^\n]*\b'+sym+r'(?=[,<(])')
   hits=[x for x in ids if re.search(regex,x['Kernel Name'])]
   if not hits:raise ValueError('No intendedfamilyobserved '+n+'/'+m+'/'+sym)
   proof={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':'timed_scientific_family','exact_stage_callsite':st['source_callsite'],'reason':INFO[n]['why'],'input_parameter_state_qualification':INFO[n]['params'],'scientific_family':INFO[n]['family'],'precision_and_non_scalar_work_qualification':INFO[n]['precision'],'evidence':evidence,'actual_profile_source_bindings':selected[-1]['plan'],'additional_profile_source_bindings':selected[-1]['provenance'],'selected_result':r['reference_result'],'selected_config':r['raw_config'],'selected_profile_manifest':selected[-1]['manifest'],'selected_counter_derivation':selected[-1]['derivation'],'profile_target_command':plan['profile_target_command'],'regex_family_correspondence':{'regex':regex,'matched_observed_identities':hits,'meaning':'Exactsource-dispatched semanticfamily ownership; no launch/casecount or dynamicphaseproof.'},'sample_phase':'unknown','sample_in_primary_timed_invocation':None,'retained_trial':None,'limitations':['Samefamilymayexecuteuntimedwarmupandtimedclosure; selectedfirstsamplephase unproved.','No completeROI/orderedlaunchtrace/elapsedaggregate/causal orSASS/tensorclaim.','Sourceprecisionqualifiers derive actualtyped arithmetic/literals/functions, notartifactstoragealone.']}
   updates={'model_kernel_regex':regex,'semantic_family_in_primary_ROI':True,'ROI_membership_status':'source_timed_closure_inclusion_proved','ROI_membership_proof':proof};stagepatch.append({'stage_index':si,'stage_id':st['stage_id'],'expected_original_stage_canonical_sha256':canonical(st),'updates':updates});revised[si].update(updates)
   for oi,other in enumerate(r['stages']):
    if oi in[i for i,_ in owners]or other.get('callable_candidates')!=st.get('callable_candidates'):continue
    oldpat=other.get('model_kernel_regex')
    if oldpat and any(re.search(oldpat.replace('\\\\','\\'),x['Kernel Name'])for x in hits):
     up={'model_kernel_regex':None,'kernel_family_alias_of':st['stage_id']};stagepatch.append({'stage_index':oi,'stage_id':other['stage_id'],'expected_original_stage_canonical_sha256':canonical(other),'alias_only':True,'updates':up,'alias_proof':{'reason':'Exactsame source-resolved callbackfamily as provedowner; duplicatewarmup/dependency/legacyCLIcallsite is not separatecounterfamily. Preservemembership/status unchanged onalias; nofalseuntimedexclusioninference.','owner_stage_id':st['stage_id'],'callable_candidates':st['callable_candidates'],'evidence':evidence}});revised[oi].update(up)
  mappings=[]
  for identity in ids:
   hits=[st['stage_id']for st in revised if st.get('model_kernel_regex')and re.search(st['model_kernel_regex'],identity['Kernel Name'])]
   if len(hits)>1:raise ValueError('Ambiguousfamilyownership '+n+'/'+m+' '+str(hits))
   mappings.append({'identity':identity,'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  patches.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':n,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canonical(r),'stage_patches':stagepatch,'unchanged_record_qualifications':{k:r[k]for k in('full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind')if k in r},'selected_observation_matching':mappings})
# Sources rechecked after proofconstruction; parent/selectedjoins rechecked laterunderLOCK_SH.
for path,b in sources.items():
 if file(path)['sha256']!=b['sha256']:raise ValueError('Sourcechanged during review')
out={'schema_version':1,'scope':'Private scientificROI source-membership proposal for only axhelm,bicgstab,ccsd-trpdrv,clenergy,chemv. Existing25adoptednames untouched.','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':patches,'unresolved_cases':unresolved,'merging_policy':'Root revalidates LOCK_EX parentcompressed/rawSHA,canonicalrecord/stageSHA,selectedcapture/config/result/sourcejoins. Preserveoldauthority/history; alias_onlychanges regex/aliasonly. Observedphaseunknown, nofullROI/causalclaim. BiCGStab remainsunresolved kernelassociation.','GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False,'shared_effort_not_translation':True}
(O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'selected_models':len(selected),'patchrecords':len(patches),'inclusions':sum(x['updates'].get('ROI_membership_status')=='source_timed_closure_inclusion_proved'for r in patches for x in r['stage_patches']),'aliases':sum(bool(x.get('alias_only'))for r in patches for x in r['stage_patches']),'unresolved_models':len(unresolved),'sources':len(sources),'proposal_sha256':sha((O/'proposed_stage_patches.json').read_bytes())}))
