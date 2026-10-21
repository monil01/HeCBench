"""CPU-only existing-evidence/source annotations; no study imports/execution."""
from pathlib import Path
import collections,copy,fcntl,gzip,hashlib,json,re,datetime
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';D=Path(__file__).resolve().parent
NAMES=['all-pairs-distance','amgmk','asta','adv','bh'];MODELS=['cuda','cuda_jl','ka','jacc']
def sh(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return sh(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
blobs={};ledgers={};records={};profiles={};parentpaths=set()
def keep(p):
 b=p.read_bytes();blobs[str(p)]=b;return b
# Short read-only metadata snapshot; no lock file creation/truncation/writes.
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for path in sorted((S/'roi_reviews').glob('*.json.gz')):
  packed=path.read_bytes();doc=json.loads(gzip.decompress(packed));chosen=[(i,x)for i,x in enumerate(doc['records'])if x['benchmark']in NAMES]
  if chosen:
   blobs[str(path)]=packed;parentpaths.add(str(path))
   for i,x in chosen:records[(x['benchmark'],x['model'])]=(path,i,x)
 for b in NAMES:
  p=S/'benchmarks'/(b+'.json');ledgers[b]=json.loads(keep(p))
  for m in MODELS:
   state=ledgers[b]['models'][m]
   if state.get('profiling_status')!='profile_process_completed'or not state.get('profile_ids'):continue
   folder=R/state['profile_ids'][0];manifest=json.loads(keep(folder/'manifest.json'));files={}
   for name in ['plan.json','profile_provenance.json','derived.json']:
    entry=next(x for x in manifest['files']if x['original_name']==name);p=R/entry['path'];z=keep(p);raw=gzip.decompress(z)
    assert sh(z)==entry['compressed_sha256']and len(z)==entry['compressed_bytes']and sh(raw)==entry['uncompressed_sha256']and len(raw)==entry['uncompressed_bytes']
    files[name]=json.loads(raw)
   profiles[(b,m)]=(folder,files)
   for n in ['config.json','result.json']:keep(S/'selected_contracts'/b/(n+'.gz'))
 fcntl.flock(lock,fcntl.LOCK_UN)
# Retain exact private snapshot bytes after releasing shared lock.
snapshot=D/'input_snapshot';snapshot.mkdir(exist_ok=True)
for path,data in blobs.items():
 p=snapshot/(sh(path.encode())[:16]+'_'+Path(path).name)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)

def source_evidence(path,bindings,terms=()):
 path=Path(path).resolve();keys=[k for k in bindings if Path(k).resolve()==path];assert keys,('not captured',str(path))
 raw=path.read_bytes();digest=sh(raw);assert all(bindings[k]==digest for k in keys),('current source differs',str(path))
 lines=raw.decode().splitlines();sel=set()
 needles=('time_ns','steady_clock','run();','run!(','launch!','launch(','observer(','observe_params','reset();','reset!','function factory','function build','function study','function run','for (stage','for s in','for(int s=','device_synchronize','CUDA.synchronize','cudaDeviceSynchronize','gpu_relax_batch!',*terms)
 for i,line in enumerate(lines):
  if any(x in line for x in needles):sel.update(range(max(0,i-1),min(len(lines),i+2)))
 return {'path':str(path),'sha256':digest,'captured_binding_keys':keys,'timer_lines':[{'line':i+1,'text':lines[i]}for i in sorted(sel)],'role':'Exact current source bytes already bound before/after selected capture; no retrospective identity backfill.'}

FAMILIES={
'all-pairs-distance':{'cuda':[('k3',0)],'cuda_jl':[('apd_kernel_',0)],'ka':[('gpu_apd_ka_',0)],'jacc':[('apd_jacc_',0)]},
'amgmk':{'cuda':[('relax',3),('observe_params',2)],'cuda_jl':[('relax_kernel_',0),('observe_kernel_',1)],'ka':[('gpu_relax_kernel_',0),('gpu_observe_kernel_',1)],'jacc':[('relax_kernel_',0),('observe_kernel_',1)]},
'asta':{'cuda':[('PTTWAC_soa_asta',0)],'cuda_jl':[('kernel_',0)],'ka':[('gpu_kernel_',0)],'jacc':[('kernel_',0)]},
'adv':{'cuda':[('advCubatureHex3D',1)],'cuda_jl':[('adv_kernel_',0)],'ka':[('gpu_adv_kernel_',0)],'jacc':[('adv_kernel_',0)]},
'bh':{'cuda':[(x,i)for i,x in enumerate(['InitializationKernel','BoundingBoxKernel','ClearKernel1','TreeBuildingKernel','ClearKernel2','SummarizationKernel','SortKernel','ForceCalculationKernel','IntegrationKernel'])],'cuda_jl':[(f'bh_{x}_',i)for i,x in enumerate(['init','bbox','clear1','build','clear2','summarize','sort','force','integrate'])],'ka':[(f'gpu_ka_bh_{x}_',i)for i,x in enumerate(['init','bbox','clear1','build','clear2','summarize','sort','force','integrate'])],'jacc':[(f'jacc_bh_{x}_',i)for i,x in enumerate(['init','bbox','clear1','build','clear2','summarize','sort','force','integrate'])]}}
NOTES={
'all-pairs-distance':{'timer':'Named224instances×4096attributes k3 contract. Native launch lambda calls only k3, repeated inside start/device-completion interval after reset. Julia apd_study builds model factory and passes its closure to OwnedStudy.run_trials!, whose repeated launch is timed after reset! and before output readback. CUDA.jl study.jl suppresses legacy main and substitutes dimensions before include_string. Same family also warms up; sampled phase remains unknown. Legacy native k1/k2 are distinct unselected implementations, not aliases or proved setup kernels.','precision':'UInt8/char categorical inputs and Int32 mismatch counts (Hamming-like symbol differences), not floating Euclidean distance. Native k3 has CUB integer BlockReduce and lane0 store, no atomicAdd (k1 does have atomicAdd but is unselected). Julia one workitem/block per pair sequentially accumulates Int32 without atomics.','boundary':'Output fill/cudaMemset is before each timed loop; input upload/allocation, independent CPU counts, readback and artifacts are outside.'},
'amgmk':{'timer':'Primary sum comprises separately timed CPU CSR matvec, GPU in-place asynchronous relaxation batch, and CPU axpy (R500,50cube). gpu_relax_batch!/native launch loop is inside its own timer through device-wide completion. observe_params/observe_kernel is called before all three clocks to echo N/NX/NY/NZ/R; it is setup/validation, not a timed solver kernel. Equilibrium and warmup use the same relaxation family outside trials.','precision':'Actual selected state/CSR values/RHS use Float64/double with Int32 indices. In-place nonsynchronized neighbor relaxation retains scheduling-dependent raw states; property-equivalence and weighted residual/range contracts remain, not identical raw-state claims. No Jacobi/redblack or atomic substitute is inferred. Observer only stores Int32 parameter metadata.','boundary':'Host/device buffers reset and parameter observation/readback precede timers. Device input/output/property captures, guards, weighted residual and equilibrium checks and file writing are outside. CPU matvec/axpy are inside primary sum but have no GPU kernel-family samples. Retained exact path-only overlay text is verified against captured copySHA; original scratch copies no longer exist.'},
'asta':{'timer':'ASTA is atomic cycle-following in-place matrix transpose, not a PDE operator. Selected study(run!) invokes one!/native one lambda: copy input and reset head/finished, synchronize, start clock, invoke transpose dispatch, synchronize, end. Five payload lengths1,32,65,129,256 use the same family; each case100launches/checks and untimed10warmups remain source context only, not inferred profiler multiplicities.','precision':'Float32 payload values are loaded/copied, not used for floating-point arithmetic. Int32 queue head/finished flags, atomic_add/atomic_xchg and cycle/permutation integer calculations plus block barriers control work. KA/JACC reuse explicit CUDA device atomics; JACC shared8B controls are separate from payload precision.','boundary':'H2D reset, runtime fills, output/flag/head readbacks and exact permutation checking/file writes are outside each perlaunch timer. The same source family appears in warmup.'},
'adv':{'timer':'Selected two-element/two-input 3D cubature advection study creates launch closures before trials; each closure is repeatedly called between the source start clock and device-completion boundary. Native closure is one advCubatureHex3D with2blocks16×16. Julia build closure iterates all seven Val(1)..Val(7) tensor interpolation/derivative/flux/backprojection stages using the same specialized adv_kernel family.','precision':'Actual fixtures/state/workspaces are Float64/native dfloat double. Stage Val branches and bounds select extents3E×1024,2048,4096,4096,512,512,512; no atomics are in the selected stage body. Integer indexing is not counted FP work; source family grouping does not fuse seven stages into one observed launch.','boundary':'Input uploads, prepare/workspace allocations and explicit launch warmups precede timing; full CPU/source-gold comparison/readback and artifacts follow. Output buffers are overwritten by each staged pipeline, with no per-repeat reset inside the clock.'},
'bh':{'timer':'Selected bh_study/Case.run full octree pipeline is called after case reset and timer start, followed by device completion. Initialization belongs inside run, then each requested step dispatches bbox,clear1,tree-build,clear2,summarize,sort,force,integrate. Cases use(256,1),(256,3),(1024,1),(1024,3),(2048,1); warmup calls same run outside trials. All nine source families are timed members, including integer tree setup.','precision':'Positions/mass/velocity/acceleration use Float32/float4, while topology/child/count/sort/step and atomicCAS/add/sub/inc counters are Int32/UInt32. Tree readiness/locks, volatile loads, barriers, warp votes/shuffles and rsqrt are explicit backend operations. Force distinguishes leaf vs opening-threshold tests and step>0 velocity correction; integration uses the source Float32 dt/halfdt. Scalar add/mul/FMA counters exclude integer, atomic/control and SFU costs.','boundary':'External position/velocity/acceleration copies and allocation-guard setup precede timers. Internal clear/init/tree construction are inside. Full state readback, topology/permutation/root-mass checks, independent force comparison and guards/artifacts occur after completion. Raw topology uniqueness or whole-workflow bottlenecks are not inferred.'}}

patches=[];captures=[];held=[];summaries=[];sourcechecks={}
for b in NAMES:
 for m in MODELS:
  key=(b,m)
  if key not in profiles:held.append({'benchmark':b,'model':m,'reason':'no selected completed capture'});continue
  path,index,record=records[key];folder,files=profiles[key];plan=files['plan.json'];prov=files['profile_provenance.json'];derived=files['derived.json'];rawdoc=gzip.decompress(blobs[str(path)])
  try:
   assert prov['status']=='profile_process_completed'and prov['plan_sha256']==sh(gzip.decompress(blobs[str(folder/'plan.json.gz')]))
   assert record['reference_result']['sha256']==plan['reference_result_sha256']and record['raw_config_sha256']==plan['raw_config_sha256']
   for name,expected in [('config.json',plan['raw_config_sha256']),('result.json',plan['reference_result_sha256'])]:assert sh(gzip.decompress(blobs[str(S/'selected_contracts'/b/(name+'.gz'))]))==expected
   bindings=dict(plan['execution_bindings'])
   for k,v in prov.get('additional_current_profile_input_bindings',{}).items():assert k not in bindings or bindings[k]==v;bindings[k]=v
   parent=R/'src'/(b+'-julia');child=parent/(b+'-julia-'+{'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}.get(m,''))
   # Include all scientific .cu/.jl/.h sources already bound to this capture,
   # plus exact library-free shared timing harness when it is in the map.
   candidates=[]
   for name in bindings:
    p=Path(name)
    if p.suffix in('.cu','.jl','.h')and p.is_file()and(str(p.resolve()).startswith(str(parent))or str(p.resolve()).startswith(str(R/'src'/(b+'-cuda')))or p.name=='OwnedStudy.jl'):candidates.append(p.resolve())
   sources=[source_evidence(p,bindings)for p in sorted(set(candidates))]
   assert sources
   for e in sources:sourcechecks[e['path']]=e['sha256']
   overlay=[]
   for entry in plan.get('overlay_provenance',{}).get('profile_source_patches',[]):
    assert sh(entry['text'].encode())==entry['copy_sha256']and bindings[entry['copy']]==entry['copy_sha256']and bindings[entry['original']]==entry['original_sha256']
    overlay.append({'actual_executed_source_path':entry['copy'],'actual_executed_sha256':entry['copy_sha256'],'retained_in_selected_plan':True,'retained_text':entry['text'],'original_bound_path':entry['original'],'original_sha256':entry['original_sha256'],'original_overlay_file_currently_exists':Path(entry['copy']).is_file(),'qualification':'Executed-copy bytes retained exactly in immutable captured plan.overlay_provenance.profile_source_patches.text; not reconstructed by rewriting original source.'})
   capture={'benchmark':b,'model':m,'manifest':{'path':str(folder/'manifest.json'),'sha256':sh(blobs[str(folder/'manifest.json')])},'plan':{'path':str(folder/'plan.json.gz'),'sha256':sh(blobs[str(folder/'plan.json.gz')])},'provenance':{'path':str(folder/'profile_provenance.json.gz'),'sha256':sh(blobs[str(folder/'profile_provenance.json.gz')])},'derivation':{'path':str(folder/'derived.json.gz'),'sha256':sh(blobs[str(folder/'derived.json.gz')])},'config_sha256':plan['raw_config_sha256'],'result_sha256':plan['reference_result_sha256'],'configuration_id':plan['configuration_id'],'selected_profile_id':str(folder.relative_to(R)),'captured_overlay_source_evidence':overlay}
   proposed=[];operations=[];canonical_ids=set();family_entries=FAMILIES[b][m]
   for family,stageindex in family_entries:
    old=record['stages'][stageindex];canonical_ids.add(old['stage_id']);pattern=(r'^_parallel_(?:for|reduce)_cuda[^\n]*\b'+re.escape(family)+r'(?=[,<(])')if m=='jacc'else r'\b'+re.escape(family)+r'(?=[(<])'
    sampled=[x['identity']for x in derived['records']if re.search(pattern,x['identity']['Kernel Name'])]
    assert sampled,('no matching saved family',b,m,family)
    excluded=b=='amgmk'and'observe'in family
    explanation=NOTES[b]
    proof={'proof_kind':'source_timer_to_callback_to_exact_family_not_runtime_phase','callsite_kind':'setup_or_untimed_parameter_observation'if excluded else'timed','timed_closure_boundaries':sources,'timer_to_dispatch_explanation':explanation['timer'],'precision_branch_atomic_integer_context':explanation['precision'],'reset_setup_and_audit_boundary':explanation['boundary'],'captured_overlay_source_evidence':overlay,'selected_profile_id':capture['selected_profile_id'],'selected_config_sha256':plan['raw_config_sha256'],'selected_result_sha256':plan['reference_result_sha256'],'selected_configuration_id':plan['configuration_id'],'sample_phase':'unknown_not_proved_from_source_membership','whole_elapsed_ROI_claim':False,'launch_multiplicity_extrapolation':False,'causal_bottleneck':'unknown'}
    updates={'model_kernel_regex':pattern,'regex_status':'single_bound_family_identity_source_membership_not_sample_phase','semantic_family_in_primary_ROI':not excluded,'ROI_membership_status':'source_setup_exclusion_proved'if excluded else'source_timed_closure_inclusion_proved','ROI_membership_proof':proof}
    operations.append({'stage_index':stageindex,'stage_id':old['stage_id'],'expected_original_stage_canonical_sha256':canonical(old),'updates':updates})
   # Disable only exact same-family duplicate callsites. Unselected native k1/k2
   # are distinct families, not aliases, and remain unresolved.
   for i,old in enumerate(record['stages']):
    if old['stage_id']in canonical_ids:continue
    matches=[]
    for family,idx in family_entries:
     if re.search(re.escape(family)+r'(?![A-Za-z0-9_])',str(old.get('model_kernel_regex',''))):matches.append(record['stages'][idx]['stage_id'])
    if len(matches)==1:operations.append({'stage_index':i,'stage_id':old['stage_id'],'expected_original_stage_canonical_sha256':canonical(old),'alias_only':True,'updates':{'model_kernel_regex':None,'regex_status':'duplicate_same_device_family_warmup_or_legacy_callsite_not_an_independent_sample','kernel_family_alias_of':matches[0]}})
   opmap={x['stage_id']:x['updates']for x in operations}
   for old in record['stages']:proposed.append({**old,**opmap.get(old['stage_id'],{})})
   joins=[]
   for x in derived['records']:
    ids=[s['stage_id']for s in proposed if s.get('model_kernel_regex')and re.search(s['model_kernel_regex'],x['identity']['Kernel Name'])];assert len(ids)<=1,('ambiguous family',b,m,x['identity']['Kernel Name']);joins.append({'identity':x['identity'],'matched_stage_ids':ids,'runtime_first_sample_phase':'unproved','source_family_membership_only':True})
   patches.append({'parent_review':{'path':str(path),'compressed_sha256':sh(blobs[str(path)]),'uncompressed_sha256':sh(rawdoc)},'record_index':index,'record_key':{'benchmark':b,'model':m,'configuration_id':record['configuration_id']},'expected_original_record_canonical_sha256':canonical(record),'unchanged_record_qualifications':{k:v for k,v in record.items()if k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'stage_patches':operations,'selected_observation_matching':joins})
   captures.append(capture);summaries.append({'benchmark':b,'model':m,'inclusions':sum(x['updates'].get('semantic_family_in_primary_ROI')is True for x in operations),'exclusions':sum(x['updates'].get('semantic_family_in_primary_ROI')is False for x in operations),'aliases':sum(x.get('alias_only',False)for x in operations),'selected_profile_id':capture['selected_profile_id']})
  except Exception as e:held.append({'benchmark':b,'model':m,'reason':str(e)})
# Repeat the selected/review/frozen-member snapshot comparison under the same
# existing read-only shared lock, after source analysis. Hold on any parent drift.
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 drift=[path for path,raw in blobs.items()if Path(path).read_bytes()!=raw]
 fcntl.flock(lock,fcntl.LOCK_UN)
source_drift=[path for path,h in sourcechecks.items()if sh(Path(path).read_bytes())!=h]
packet={'schema_version':1,'activity':'private_source_membership_science_infra_v3','benchmarks':NAMES,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False,'selected_capture_bindings':captures,'patches':patches,'held':held,'summary':summaries,'consistency':{'initial_and_final_snapshot':'Short existing metadata LOCK_SH opened read-only; released before source analysis. Parent documents, selected ledgers and frozen members rechecked after analysis.','metadata_drift':drift,'source_drift':source_drift,'root_merge_requires_new_LOCK_EX_validation':True},'no_shared_edits_GPU_build_import_tests':True}
if drift or source_drift:
 packet['status']='HOLD_INPUT_DRIFT_NO_MERGE';packet['patches']=[]
else:packet['status']='PRIVATE_PROPOSAL_READY_FOR_ROOT_MERGE_REVIEW'
dump(D/'proposed_stage_patches.json',packet);dump(D/'source_bindings.json',sourcechecks)
print(json.dumps({'status':packet['status'],'models':len(patches),'inclusions':sum(x['inclusions']for x in summaries),'exclusions':sum(x['exclusions']for x in summaries),'aliases':sum(x['aliases']for x in summaries),'held':held,'metadata_drift':drift,'source_drift':source_drift}))
