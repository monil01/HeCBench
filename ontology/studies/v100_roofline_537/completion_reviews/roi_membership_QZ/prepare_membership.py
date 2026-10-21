"""Private source-membership proposals only; no measurement or shared-file writes."""
import collections,gzip,hashlib,json,re
from pathlib import Path
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';OUT=Path(__file__).resolve().parent
NAMES=['quant3MatMul','quantBnB','quantAQLM','randomAccess','svd3x3','wmma'];MODELS=['cuda','cuda_jl','ka','jacc']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rawsha(b):return hashlib.sha256(b).hexdigest()
def readfrozen(slot,name):
 manifest=json.loads((slot/'manifest.json').read_bytes());e=next(x for x in manifest['files']if x['original_name']==name);p=R/e['path'];packed=p.read_bytes();raw=gzip.decompress(packed)
 if rawsha(packed)!=e['compressed_sha256']or rawsha(raw)!=e['uncompressed_sha256']or len(packed)!=e['compressed_bytes']or len(raw)!=e['uncompressed_bytes']:raise RuntimeError('frozen member mismatch')
 return json.loads(raw),{'path':str(p),'compressed_sha256':rawsha(packed),'raw_sha256':rawsha(raw)}
def snippet(p):
 lines=p.read_text().splitlines();hits=[i for i,l in enumerate(lines)if any(t in l for t in ['function factory','function run_svd','function study(','function study_main','function quant','run_trials!','auto launch=','void launch(','double run()','for trial','for(int trial','for (int trial','for _','start=','start =','auto start=','s.launch','for rep','run!(','run();','for(int b=','for(int r=','launch_svd','study(','study(factory)','wmma_study','quant_study','quantbnb_study','quant3_study'])];keep=set()
 for i in hits:keep.update(range(max(0,i-1),min(len(lines),i+3)))
 return [{'line':i+1,'text':lines[i]}for i in sorted(keep)]
def source(p,bindings):
 p=p.resolve();keys=[k for k in bindings if Path(k).resolve()==p]
 if not keys:raise RuntimeError('source not actually captured/bound: '+str(p))
 h=sha(p)
 if any(bindings[k]!=h for k in keys):raise RuntimeError('source current bytes differ from captured profile '+str(p))
 return{'path':str(p),'sha256':h,'profile_binding_keys':keys,'line_evidence':snippet(p)}
FAMILIES={
 'quant3MatMul':{'cuda':['VecQuant3MatMulKernel'],'cuda_jl':['quant3_kernel_'],'ka':['gpu_quant3_ka_'],'jacc':['quant3_jacc_']},
 'quantBnB':{'cuda':['kQuantize'],'cuda_jl':['kquantize_'],'ka':['gpu_quantbnb_ka_'],'jacc':['quantbnb_jacc_']},
 'quantAQLM':{'cuda':['Code1x16MatVec'],'cuda_jl':['quant_kernel'],'ka':['gpu_quant_kernel'],'jacc':['quant_kernel']},
 'randomAccess':{'cuda':['initTable','update'],'cuda_jl':['init_','update_'],'ka':['gpu_init_','gpu_update_'],'jacc':['init_','update_']},
 'svd3x3':{'cuda':['svd3_SOA'],'cuda_jl':['factor_kernel_'],'ka':['gpu_factor_kernel_'],'jacc':['factor_kernel_']},
 'wmma':{'cuda':['gemm_impl0','gemm_impl1'],'cuda_jl':['gemm_impl0','gemm_impl1'],'ka':['gpu_gemm_impl0','gpu_gemm_impl1'],'jacc':['gemm_impl0','gemm_impl1']}}
NOTES={
 'quant3MatMul':'Selected study constructs operation! with quant3 factory. launch! zeroes output then invokes operation!. OwnedStudy.run_trials! brackets the repeated launch! closure with time_ns and CUDA.synchronize; input allocation/CPU gold/readback/write occur outside. Native study launch lambda also includes cudaMemset then VecQuant3MatMulKernel, repeated inside chrono interval. Thus quant kernel family belongs to primary timed closure; output zeroing also belongs, so this is NOT a complete-ROI representation. Faster kernel path in included legacy native main is not invoked by selected adapter.',
 'quantBnB':'Selected factory returns only the quantization kernel closure. quantbnb_study passes it to OwnedStudy.run_trials!: reset/warmup precede trial timer; repeated launch! is inside start/time_ns and synchronization; CPU codebook parsing/gold/readback/conversion/I/O outside. Native selected adapter launch lambda kQuantize is repeated inside chrono interval. A warmup uses the same kernel family, so first observed sample phase is not thereby proved timed.',
 'quantAQLM':'Selected native launch(c,b) invokes Code1x16MatVec; study loops both selected cases, four vectors and R repeats with start clock before launch and device completion before elapsed accumulation. Julia quant_study creates case launch=factory and applies the same b/repeat timed closure before times.push(elapsed/R). Factory resolves to quant_kernel/gpu_quant_kernel or JACC quant_kernel. Actual input/guard echoes and dense quality/audit are after the timer. Corrected-reference extent/stride qualification and old failure remain unchanged.',
 'randomAccess':'Selected run!/native run lambda launches BOTH init family and update family. study brackets repeat run! with clock and CUDA completion. Therefore initTable/init!/gpu_init!/JACC init! is timed, not setup-only, despite also being used by warmup. CUDA.zeros initial runtime fill is a separate setup family not promoted by this patch; init and update row membership does not assign runtime-fill sample phase or aggregate launch counts.',
 'svd3x3':'Selected launch_svd resolves to the factorization family. Julia run_svd synchronizes, starts time_ns, repeats launch_svd, synchronizes and divides elapsed by SVD_R; study_main sums run_svd over six cases after untimed inputs/outputs reset and readback. Native Work.run likewise times only repeated svd3_SOA and main sums each case run. First calls are explicit warmups using the same family. Scientific property checker/capture is outside this timed family; nonunique factorization/qualified reference contract remains unchanged.',
 'wmma':'Selected native launch(State) dispatches impl0/impl1 by case impl; selected Julia factory does the same and stores closure as s.launch. Native/Julia trial loop resets output before clock, repeats launch for each selected case, completes device and appends elapsed; sums cases. Both kernel families therefore appear in primary timed closure. Thirty preparatory launches and native out-of-contract diagnostic can use same family outside timer: first-invocation phase remains UNKNOWN. No 11case/repeat launch extrapolation or tensor conversion is authorized by this source proof.'}
reviews={};parentdocs={}
for rp in sorted((S/'roi_reviews').glob('*.json.gz')):
 packed=rp.read_bytes();raw=gzip.decompress(packed);doc=json.loads(raw)
 for rec in doc['records']:
  if rec['benchmark']in NAMES:reviews[(rec['benchmark'],rec['model'])]=(rec,{'path':str(rp),'compressed_sha256':rawsha(packed),'raw_sha256':rawsha(raw),'record_sha256':rawsha(json.dumps(rec,sort_keys=True,separators=(',',':')).encode())})
patches={};aliases={};held=[];evidence={};ledgers={}
for b in NAMES:
 lp=S/'benchmarks'/(b+'.json');blob=lp.read_bytes();ledgers[str(lp)]=rawsha(blob);ledger=json.loads(blob)
 for model in MODELS:
  rec,parent=reviews[(b,model)];state=ledger['models'][model];ids=state.get('profile_ids',[])
  if not ids:held.append({'benchmark':b,'model':model,'reason':'no selected profile'});continue
  slot=R/ids[0]
  try:
   if state.get('profiling_status')!='profile_process_completed':raise RuntimeError('selected failed parent: source proof not promoted; retain failed evidence')
   plan,planproof=readfrozen(slot,'plan.json');prov,provproof=readfrozen(slot,'profile_provenance.json')
   if planproof['raw_sha256']!=prov['plan_sha256']:raise RuntimeError('plan/prov mismatch')
   if rec['reference_result']['sha256']!=plan['reference_result_sha256']or rec['raw_config_sha256']!=plan['raw_config_sha256']:raise RuntimeError('candidate/selected contract mismatch')
   bd=dict(plan['execution_bindings'])
   for k,h in prov.get('additional_current_profile_input_bindings',{}).items():
    if k in bd and bd[k]!=h:raise RuntimeError('contradictory captured source binding')
    bd[k]=h
   parentdir=R/'src'/(b+'-julia');child=parentdir/(b+'-julia-'+{'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}.get(model,''))
   files=[parentdir/'study_native.cu']if model=='cuda'else[child/('study.jl'if b in ['quant3MatMul','quantBnB']and model=='cuda_jl'else'main.jl'),parentdir/'study_common.jl']
   if model!='cuda'and b in ['quant3MatMul','quantBnB']:files.append(R/'src/_portable_study/OwnedStudy.jl')
   # Native family definition is bound too; proofs of wrapper dispatch retain it.
   if model=='cuda':
    definitions={'quant3MatMul':R/'src/quant3MatMul-cuda/kernels.h','quantBnB':R/'src/quantBnB-cuda/main.cu','quantAQLM':parentdir/'native_corrected/main.cu','randomAccess':R/'src/randomAccess-cuda/main.cu','svd3x3':parentdir/'native_corrected/main.cu','wmma':R/'src/wmma-cuda/main.cu'}
    if definitions[b].is_file():files.append(definitions[b])
   sources=[source(p,bd)for p in files]
   key=b+'/'+model;evidence[key]={'selected_profile_id':ids[0],'selected_status':state['profiling_status'],'ledger':{'path':str(lp),'sha256':rawsha(blob)},'plan':planproof,'provenance':provproof,'manifest_sha256':sha(slot/'manifest.json'),'sources':sources,'candidate_parent_review':parent,'config_result_hashes':{'config':plan['raw_config_sha256'],'result':plan['reference_result_sha256']},'profile_build_command':plan['build_command'],'profile_target_command':plan['profile_target_command'],'scope':'Source semantic family membership only; no accepted data admission if parent failed; no current-source historical backfill. Current source bytes match already captured hashes.'}
   kerneldata=slot/'derived.json.gz';names=[]
   if kerneldata.exists():
    der,derproof=readfrozen(slot,'derived.json');names=[x['identity']['Kernel Name']for x in der['records']];evidence[key]['derived']=derproof
   used=set()
   for family in FAMILIES[b][model]:
    candidates=[s for s in rec['stages']if re.search(re.escape(family)+r'(?![A-Za-z0-9_])',str(s.get('model_kernel_regex','')))]
    if not candidates:raise RuntimeError('no existing stage family '+family)
    canonical=candidates[0];stageid=canonical['stage_id'];used.add(stageid)
    pattern=(r'^_parallel_for_cuda(?:_[A-Z]+)?\([^\n]*\b'+re.escape(family)+r'(?=[,<(])')if model=='jacc'else(r'\b'+re.escape(family)+r'(?=[(<])')
    actual=[n for n in names if re.search(pattern,n)]
    if names and not actual:raise RuntimeError('regex does not match captured function family '+family)
    proof={'source_profile_evidence_key':key,'explanation':NOTES[b],'call_chain_kind':'selected_timer_to_model_factory_to_device_kernel_family','source_membership_only':True,'first_invocation_sample_phase':'unknown_not_proved_inside_primary_timed_invocation','complete_ROI':False,'launch_count_extrapolation':False,'causal_bottleneck':'unknown','saved_kernel_family_matches':actual,'regex_match_is_identity_annotation_not_new_test':True}
    patches[stageid]={'benchmark':b,'model':model,'stage_id':stageid,'parent_review':parent,'previous_stage_sha256':rawsha(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()),'updates':{'model_kernel_regex':pattern,'regex_status':'source_bound_single_family_symbol_identity_sample_phase_unproved','semantic_family_in_primary_ROI':True,'ROI_membership_status':'source_timed_closure_inclusion_proved','ROI_membership_proof':proof}}
    for duplicate in candidates[1:]:
     if duplicate['stage_id']in used:continue
     aliases[duplicate['stage_id']]={'benchmark':b,'model':model,'stage_id':duplicate['stage_id'],'parent_review':parent,'previous_stage_sha256':rawsha(json.dumps(duplicate,sort_keys=True,separators=(',',':')).encode()),'updates':{'model_kernel_regex':None,'regex_status':'duplicate_source_callsite_not_independent_kernel_family','family_alias_of':stageid},'reason':'Preserve original candidate record; disable overlapping family regex only. Warmup/legacy callsite is not independently mapped to a timed sample. Canonical same function family has timer-chain proof.'}
   # Exact post-proposal uniqueness on saved observations; descriptive CPU join only.
   proposed=[]
   for stage in rec['stages']:
    update=patches.get(stage['stage_id'],aliases.get(stage['stage_id'],{})).get('updates',{})
    proposed.append({**stage,**update})
   evidence[key]['saved_identity_membership_join']=[{'kernel_name':n,'matching_stages':[s['stage_id']for s in proposed if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],n)]}for n in names]
   if any(len(x['matching_stages'])>1 for x in evidence[key]['saved_identity_membership_join']):raise RuntimeError('ambiguous family mapping after proposal')
  except Exception as err:
   held.append({'benchmark':b,'model':model,'reason':str(err)})
   # Drop partial proposals on a model if a binding/identity review fails.
   for target in [patches,aliases]:
    for k in list(target):
     if target[k]['benchmark']==b and target[k]['model']==model:del target[k]
for p,h in ledgers.items():
 if sha(p)!=h:raise RuntimeError('selected ledger changed during read; discard/reprepare private proposal')
packet={'schema_version':1,'activity':'source_only_ROI_membership_QZ_proposal','benchmarks':NAMES,'stage_updates':patches,'duplicate_regex_alias_updates':aliases,'evidence':evidence,'held':held,'membership_statuses_used':['source_timed_closure_inclusion_proved'],'sample_phase':'unknown for first invocation; membership is not phase or complete ROI proof','scope':'private proposal only; no shared review/ledger/tools updates, GPU/build/tests, launch extrapolation or speedup correlation','consistency':'Read exact immutable frozen member SHA/length; selected-ledger bytes rehashed unchanged after reading. Not represented as a coordinator-lock snapshot. Background SH read was waiting behind an exclusive holder. Root must revalidate selected parent/source review SHA at merge.'}
(OUT/'membership_patch.json').write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
print(json.dumps({'promoted_stages':len(patches),'duplicate_aliases':len(aliases),'covered_model_records':len({(v['benchmark'],v['model'])for v in patches.values()}),'held':held}))
