import json,gzip,hashlib,re,datetime
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;N={'laplace3d','particle-diffusion','haccmk','jacobian'}
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
  ledger=json.loads((S/'benchmarks'/f'{b}.json').read_text());mod=ledger['models'][m];
  if mod['profiling_status']!='profile_process_completed':continue
  mp=W/mod['profile_evidence_manifest'];pp=mp.parent/'plan.json.gz';plan=frozen(pp);obs=mp.parent/'derived.json.gz';observed=frozen(obs)['records']
  assert plan['configuration_id']==r['configuration_id'];assert plan['reference_result_sha256']==r['reference_result']['sha256'];assert plan['raw_config_sha256']==r['raw_config_sha256']
  evidence=[]
  for name in ['reference_result','raw_config']:
   p=Path(r[name]['path']);assert sha(p.read_bytes())==r[name]['sha256']
  base=W/'src'/f'{b}-julia'
  if m=='cuda':
   entry=base/('study_native.cu'if b=='haccmk'else'study_cuda.cu')
   evidence=[span(entry,entry.read_text().splitlines()[0],None)]
   why={'laplace3d':'Selected study main warmup precedes checked trials; repeat laplace3d launch is between steady_clock start and synchronized elapsed. Native historical_main renamed and never called.', 'particle-diffusion':'Selected study main defines launch lambda calling Simulation. After reset, timer directly invokes launch then synchronizes and computes elapsed. Same lambda also warms up before timer; legacy_main is renamed/dormant.', 'haccmk':'Selected study main defines launch lambda calling haccmk_kernel; timed repeat invokes launch between steady_clock start and synchronized elapsed. Output reset precedes timer; lambda warmup precedes checked loop; original_hecbench_main is dormant.'}[b]
  else:
   child=base/f"{b}-julia-{'cuda.jl'if m=='cuda_jl'else'KA'if m=='ka'else'JACC'}"
   entry=child/('study.jl'if m=='cuda_jl'else'main.jl')
   evidence.append(span(entry,entry.read_text().splitlines()[0],None))
   main=child/'main.jl'
   if main!=entry:evidence.append(span(main,main.read_text().splitlines()[0],None))
   common=base/'study_common.jl';evidence.append(span(common,common.read_text().splitlines()[0],None))
   if b in ['haccmk','jacobian']:
    helper=W/'src/_portable_study/OwnedStudy.jl';evidence.append(span(helper,'module OwnedStudy',None))
   why={'laplace3d':'Entry passes study_prepare into study_main; prepare binds named laplace kernel closure. Shared study calls same launch!(src,dst) inside twenty-step loop after t0 and before synchronized elapsed; reset beforetimer and readback after. JACC parallel_for explicitly binds laplace_kernel!, KA prepare constructs named kernel. CUDAjl entry includes main under PROGRAM_FILE guard so study_main is invoked once.', 'particle-diffusion':'Study entry passes study_prepare into study_main. Returned closure explicitly binds simulation_kernel!/portable_particle!; study_main invokes launch! after t0 and before synchronized elapsed each repeat. Resets, UInt64 fill and readback outside timer. CUDAjl include_string removes exit(main(ARGS)), so legacy main launch is dormant. JACC callback is explicit parallel_for argument.', 'haccmk':'Study entry factory returns haccmk_kernel! closure, passed to hacc_study. hacc_study passes operation! to OwnedStudy.run_trials!, which executes launch! between start=time_ns and synchronized elapsed; reset and readback outside. Included main PROGRAM_FILE guard prevents legacy main call.', 'jacobian':'Actual selected argv k2,512 reaches factory with mode k2. Ternary chooses softmax_jacobian_ka2!, then launch instance captured in returned closure. jacobian_study passes operation to OwnedStudy.run_trials!, whose timed repeat invokes launch! before sync/elapsed; setup/oracle/reset/readback outside. k1 inactive kernel is not promoted.'}[b]
  for e in evidence:
   p=Path(e['path']);sources[str(p)]=file(p)
   assert {str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()}.get(str(p.resolve()))==e['sha256'],('source not bound to actual selected profile',b,m,str(p))
  # Counter names alone do not prove phase: this checks only unique semantic-family matching.
  stagepatch=[]
  for si,s in enumerate(r['stages']):
   kind='timed'
   if b=='jacobian' and si==0:continue # k1 is not selected: leave old candidate unknown, no inaccurate setup label.
   if b=='laplace3d'and m=='cuda':kind='warmup'if si==0 else 'dormant'if si>=2 else 'timed'
   if b in ['particle-diffusion','haccmk']and m in ['cuda','cuda_jl']and si==1:kind='dormant'
   status='source_timed_closure_inclusion_proved'if kind=='timed'else'source_setup_exclusion_proved'
   proof={'proof_kind':'source_callgraph_and_enclosing_timer_not_runtime_phase','callsite_kind':kind,'exact_stage_callsite':s.get('source_callsite'),'timed_closure_boundaries':[{**file(Path(e['path'])),'timer_lines':[{'line':i+1,'text':line}for i,line in enumerate(Path(e['path']).read_text().splitlines())if any(v in line for v in ['steady_clock::now()', 'start=time_ns()', 't0=time_ns()', 'run_batch!(launch!,repeats)'])]}for e in evidence],'reason':why,'evidence':evidence,'actual_profile_source_bindings':file(pp),'selected_result':r['reference_result'],'selected_config':r['raw_config'],'selected_profile_manifest':file(mp),'selected_counter_derivation':file(obs),'profile_target_command':plan['profile_target_command'],'limitations':['Same device family may execute during setup/warmup and timed loop; first sampled invocation phase is not proved.','This source membership does not prove runtime branch, launch multiplicity, complete family/elapsed ROI coverage or causal bottleneck.','No counter extrapolation or primary timing change.'],'legacy_or_warmup_regex_policy':'Only timed semantic-family candidate retains name regex; setup/dormant callsite regex disabled to avoid ambiguous name-only match. All historical stage bytes are retained by parent SHA.'}
   updates={'semantic_family_in_primary_ROI':kind=='timed','ROI_membership_status':status,'ROI_membership_proof':proof}
   if kind!='timed':updates.update(model_kernel_regex=None,regex_status='disabled_same_kernel_warmup_or_dormant_callsite_not_independent_family')
   stagepatch.append({'stage_index':si,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':canonical(s),'updates':updates})
  byid={p['stage_id']:p['updates']for p in stagepatch};revised=[dict(s,**byid.get(s['stage_id'],{}))for s in r['stages']]
  matches=[]
  for k in observed:
   name=k['identity']['Kernel Name'];hits=[x['stage_id']for x in revised if x.get('model_kernel_regex')and re.search(x['model_kernel_regex'],name)]
   assert len(hits)<=1,(b,m,name,hits)
   matches.append({'identity':k['identity'],'matched_stage_ids':hits,'runtime_first_sample_phase':'unproved','source_family_membership_only':bool(hits)})
  proposals.append({'parent_review':{'path':str(parent),**parents[str(parent)]},'record_index':ri,'record_key':{'benchmark':b,'model':m,'configuration_id':r['configuration_id']},'expected_original_record_canonical_sha256':canonical(r),'stage_patches':stagepatch,'unchanged_record_qualifications':{k:r[k]for k in ['full_ROI_coverage','steady_state_evidence','ordered_complete_launch_census_obtained','benchmark_bottleneck','coverage_kind']},'selected_observation_matching':matches})
  selected.append({'benchmark':b,'model':m,'manifest':file(mp),'plan':file(pp),'derivation':file(obs)})
out={'schema_version':1,'scope':'G-P2 source-only private proposed patches, four NEW names/11 selected modelrecords; no shared mutation','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_reviews':parents,'selected_capture_bindings':selected,'sources':list(sources.values()),'patches':proposals,'withheld_models':{'haccmk':['ka','jacc'],'jacobian':['cuda','cuda_jl','jacc']},'withheld_stage':'jacobian/ka k1 sourcecandidate remains unknown: selected argv chooses k2; no parameter-inactive stage mislabeled as setup','merging_policy':'Root must verify parent compressed/raw SHA, record/stage canonical SHA and every source binding before patching; preserve original gzip bytes and receipt; emit no overlapping review authority.','GPU_runs':0,'build_runs':0,'tests':0,'sample_phase_still_unproved':True,'elapsed_ROI_aggregate_claim':False,'causal_claim':False}
(O/'proposed_stage_patches.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'records':len(proposals),'stages':sum(len(x['stage_patches'])for x in proposals),'input_sources':len(sources),'sha256':sha((O/'proposed_stage_patches.json').read_bytes())}))
