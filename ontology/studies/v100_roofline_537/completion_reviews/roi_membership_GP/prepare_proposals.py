import json,gzip,hashlib,re,datetime
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;N={'gabor','gamma-correction','pool'}
def sha(x):return hashlib.sha256(x).hexdigest()
def file(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def span(p,start,end=None):
 t=p.read_text();i=t.index(start);j=t.index(end,i)+len(end)if end else len(t);return {**file(p),'start_line':t[:i].count('\n')+1,'end_line':t[:j].count('\n')+1,'start_byte':len(t[:i].encode()),'end_byte':len(t[:j].encode()),'snippet':t[i:j]}
def frozen(p):return json.loads(gzip.decompress(p.read_bytes()))
def canonical(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
proposals=[];sources={};parents={};selected=[]
for parent in sorted((S/'roi_reviews').glob('*.json.gz')):
 raw=gzip.decompress(parent.read_bytes());doc=json.loads(raw)
 for ri,r in enumerate(doc['records']):
  b,m=r['benchmark'],r['model']
  if b not in N:continue
  parents[str(parent)]={'compressed_sha256':sha(parent.read_bytes()),'uncompressed_sha256':sha(raw)}
  ledger=json.loads((S/'benchmarks'/f'{b}.json').read_text());mod=ledger['models'][m];assert mod['profiling_status']=='profile_process_completed'
  mp=W/mod['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';plan=frozen(pp);obs=mp.parent/'derived.json.gz';observed=frozen(obs)['records']
  assert plan['configuration_id']==r['configuration_id'];assert plan['reference_result_sha256']==r['reference_result']['sha256'];assert plan['raw_config_sha256']==r['raw_config_sha256']
  evidence=[]
  for name in ['reference_result','raw_config']:
   p=Path(r[name]['path']);assert sha(p.read_bytes())==r[name]['sha256']
  base=W/'src'/f'{b}-julia'
  if m=='cuda':
   entry=base/('study_native.cu'if b=='gamma-correction'else'study_cuda.cu');text=entry.read_text()
   for s in r['stages']:
    if Path(s['source_callsite']['path'])==entry:
     assert sha(entry.read_bytes())==s['source_callsite']['source_sha256']
   if b=='gabor':evidence=[span(entry,'int main(',None)]
   elif b=='gamma-correction':evidence=[span(entry,'int main()',None)]
   else:evidence=[span(entry,'int main(',None)]
   why={'gabor':'In selected study main, setup gabor launch precedes trial loop; timed repeat launch is after steady_clock start and before synchronization/elapsed calculation. Legacy native main is renamed by include adapter and is not called.', 'gamma-correction':'Selected study main invokes one warmup gamma kernel before trial loop; each timed gamma dispatch occurs after steady_clock start and before synchronized elapsed accumulation. H2D reset precedes timer; included legacy_main is not called.', 'pool':'Selected study main warmup precedes trial loop; timed KernelPool2DGrad loop follows steady_clock start and precedes synchronized elapsed calculation; reset precedes timer.'}[b]
  else:
   child=base/f"{b}-julia-{'cuda.jl'if m=='cuda_jl'else'KA'if m=='ka'else'JACC'}"
   entry=child/('study.jl'if m=='cuda_jl'and b in ['gabor','pool']else'main.jl')
   evidence.append(span(entry,entry.read_text().splitlines()[0],None))
   main=child/'main.jl'
   if main!=entry:evidence.append(span(main,main.read_text().splitlines()[0],None))
   common=base/'study_common.jl'
   if b=='gabor':
    evidence.append(span(common,'function study_main(',None));why='Entry calls study_main(study_prepare); prepare returns the bound gabor callable. Shared study_main invokes that same closure in timed repeat loop after t0 and before synchronize/elapsed. CUDA.zeros and warmup closure invocation precede timers. JACC callback type gabor_kernel! is explicit at parallel_for dispatch; KA prepare constructs the named kernel instance.'
   elif b=='gamma-correction':
    evidence.append(span(common,'function study(run!)',None));why='Entry defines run! to dispatch kernel! then passes it to study(run!). Shared study invokes run! between start=time_ns and synchronized elapsed calculation. H2D reset and warmup precede timer. JACC kernel! callable is an explicit parallel_for argument; KA run! constructs the named kernel.'
   else:
    evidence.append(span(common,'function run_batch!',None));why='Entry calls study_main(prepare); prepare returns the pool2d_grad closure; shared timer encloses run_batch!(launch!,repeats), whose loop calls that closure. Reset/warmup outside timer. JACC dispatch binds pool2d_grad_kernel! explicitly; KA prepare constructs named kernel. CUDAjl study adapter strips exit(main(ARGS)) so included legacy main timing site is dormant.'
  for e in evidence:
   p=Path(e['path']);sources[str(p)]=file(p)
   assert plan['execution_bindings'].get(str(p))==e['sha256'],('source not bound to actual selected profile',b,m,str(p))
  # Counter names alone do not prove phase: this checks only unique semantic-family matching.
  stagepatch=[]
  for si,s in enumerate(r['stages']):
   kind='timed'
   if m=='cuda'and si==0:kind='warmup'
   if m=='cuda'and b in ['gabor','gamma-correction']and si==2:kind='dormant'
   if m=='cuda_jl'and b=='pool'and si==1:kind='dormant'
   status='source_timed_closure_inclusion_proved'if kind=='timed'else'source_setup_exclusion_proved'
   proof={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s.get('source_callsite'),'timed_closure_boundaries':[{**file(Path(e['path'])),'timer_lines':[{'line':i+1,'text':line}for i,line in enumerate(Path(e['path']).read_text().splitlines())if any(v in line for v in ['steady_clock::now()', 'start=time_ns()', 't0=time_ns()', 'run_batch!(launch!,repeats)'])]}for e in evidence],'reason':why,'evidence':evidence,'actual_profile_source_bindings':file(pp),'selected_result':r['reference_result'],'selected_config':r['raw_config'],'selected_profile_manifest':file(mp),'selected_counter_derivation':file(obs),'profile_target_command':plan['profile_target_command'],'limitations':['Same device family may execute during setup/warmup and timed loop; first sampled invocation phase is not proved.','This source membership does not prove runtime branch, launch multiplicity, complete family/elapsed ROI coverage or causal bottleneck.','No counter extrapolation or primary timing change.'],'legacy_or_warmup_regex_policy':'Only timed semantic-family candidate retains name regex; setup/dormant callsite regex disabled to avoid ambiguous name-only match. All historical stage bytes are retained by parent SHA.'}
   updates={'semantic_family_in_primary_ROI':kind=='timed','ROI_membership_status':status,'ROI_membership_proof':proof}
   if kind!='timed':updates.update(model_kernel_regex=None,regex_status='disabled_same_kernel_warmup_or_dormant_callsite_not_independent_family')
   stagepatch.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canonical(s),'updates':updates})
  revised=[dict(s,**p['updates'])for s,p in zip(r['stages'],stagepatch)]
  matches=[]
  for k in observed:
   name=k['identity']['Kernel Name'];hits=[x['stage_id']for x in revised if x.get('model_kernel_regex')and re.search(x['model_kernel_regex'],name)]
   assert len(hits)<=1,(b,m,name,hits)
   matches.append({'identity':k['identity'],'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canonical(r),'stage_patches':stagepatch,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'manifest':file(mp),'plan':file(pp),'derivation':file(obs)})
out={'schema_version':1,'scope':'G-P source-only private proposed patches, existing 3 names/12 models; no shared mutation','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'unselected_case':'heat2d not enriched: all4 selected profile_process_failed','merging_policy':'Root must verify parent compressed/raw SHA, record/stage canonical SHA and every source binding before patching; preserve original gzip bytes and receipt; emit no overlapping review authority.','GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
(O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'records':len(proposals),'stages':sum(len(x['stage_patches'])for x in proposals),'input_sources':len(sources),'sha256':sha((O/'proposed_stage_patches.json').read_bytes())}))
