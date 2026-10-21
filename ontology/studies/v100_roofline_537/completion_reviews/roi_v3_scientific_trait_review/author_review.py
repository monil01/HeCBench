from pathlib import Path
import json,hashlib,fcntl,gzip
R=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;sha=lambda b:hashlib.sha256(b).hexdigest();canonical=lambda j:sha(json.dumps(j,sort_keys=True,separators=(',',':')).encode())
x=json.loads((O/'extracted.json').read_text());bindings=json.loads((O/'snapshot_bindings.json').read_text());sources={};captures=[];proofs={}
a=json.loads((R/'ontology/studies/v100_roofline_537/completion_reviews/roi_membership_root_merge_v3/root_adoption.json').read_text())
for v in a['input_packets']:
 p=Path(v['path']);raw=p.read_bytes();assert sha(raw)==v['sha256'];q=json.loads(raw)
 for patch in q['patches']:proofs[(patch['record_key']['benchmark'],patch['record_key']['model'])]={'packet':v,'patch_record_key':patch['record_key'],'patch_canonical_sha256':canonical(patch),'stage_proofs':[s['updates'].get('ROI_membership_proof')for s in patch['stage_patches']if s['updates'].get('ROI_membership_proof')]}
def snap(p):
 p=Path(p);raw=p.read_bytes();key=str(p);bindings[key]={'sha256':sha(raw),'bytes':len(raw)};dst=O/'bound_sources'/key.lstrip('/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw);return raw
def getproof(n,m):return proofs[(n,m)]
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for n in x:
  ledger=json.loads((R/f'ontology/studies/v100_roofline_537/benchmarks/{n}.json').read_text());contract=ledger['accepted_named_contract']
  for m,state in ledger['models'].items():
   if m not in ['cuda','cuda_jl','ka','jacc']:continue
   path=R/state['profile_evidence_manifest'];manifest=snap(path);planraw=snap(path.parent/'plan.json.gz');provraw=snap(path.parent/'profile_provenance.json.gz');p=json.loads(gzip.decompress(planraw));v=json.loads(gzip.decompress(provraw));eb={str(Path(k).resolve()):val for k,val in {**p.get('execution_bindings',{}),**v.get('additional_current_profile_input_bindings',{})}.items()}
   c={'benchmark':n,'model':m,'configuration_id':p['configuration_id'],'raw_config_sha256':p['raw_config_sha256'],'reference_result':p['reference_result'],'reference_result_sha256':p['reference_result_sha256'],'selected_profile_manifest':{'path':str(path),'sha256':sha(manifest)},'plan_compressed_sha256':sha(planraw),'plan_uncompressed_sha256':sha(gzip.decompress(planraw)),'provenance_compressed_sha256':sha(provraw),'provenance_uncompressed_sha256':sha(gzip.decompress(provraw)),'profiling_status':state['profiling_status'],'source_bindings':eb,'adopted_stage_proof':getproof(n,m)}
   assert c['configuration_id']==contract['configuration_id'];assert c['reference_result_sha256']==contract['result_sha256'];captures.append(c)
 fcntl.flock(lock,fcntl.LOCK_UN)
# Exact source excerpts for proposed display corrections, not retrospective execution-phase claims.
def source(path,lo,hi,models):
 p=R/path;raw=snap(p);ss=sha(raw);entries=[]
 for n,m in models:
  c=next(c for c in captures if c['benchmark']==n and c['model']==m);bound=c['source_bindings'].get(str(p.resolve()));entries.append({'benchmark':n,'model':m,'selected_profile_source_sha256':bound,'hash_matches':bound==ss})
 rec={'path':str(p),'sha256':ss,'start_line':lo,'end_line':hi,'snippet':'\n'.join(raw.decode().splitlines()[lo-1:hi]),'selected_source_joins':entries};assert all(e['hash_matches']for e in entries);sources[path]=rec;return path
source('src/adv-julia/adv-julia-cuda.jl/main.jl',3,12,[('adv','cuda_jl')])
source('src/bilateral-julia/bilateral-julia-cuda.jl/main.jl',4,14,[('bilateral','cuda_jl')]);source('src/bilateral-julia/bilateral-julia-cuda.jl/study.jl',1,10,[('bilateral','cuda_jl')])
source('src/bilateral-julia/bilateral-julia-KA/main.jl',4,13,[('bilateral','ka')]);source('src/bilateral-julia/bilateral-julia-JACC/main.jl',4,13,[('bilateral','jacc')])
source('src/axhelm-julia/study_common.jl',1,45,[('axhelm',m)for m in ['cuda_jl','ka','jacc']])
# Proposed current537 display overlay: no JSON Patch targeting frozen records.
patches=[]
def add(n,m,field,value,reason,src):
 rec=x[n]['implementations'][m];path=f'ontology/studies/julia_abstractions_537/implementations/{n}/{m}.json';c=next(c for c in captures if c['benchmark']==n and c['model']==m)
 patches.append({'benchmark':n,'model':m,'display_field':field,'expected_underlying_value':rec['normalized'].get(field),'proposed_display_value':value,'underlying_record':{'path':path,'sha256':bindings[path]['sha256']},'selected_configuration_id':c['configuration_id'],'selected_result_sha256':c['reference_result_sha256'],'selected_profile_manifest':c['selected_profile_manifest'],'source_evidence_keys':src,'adopted_source_proof_ref':c['adopted_stage_proof']['packet'],'reason':reason,'authority':'current537_display_only_not_frozen_analysis_or_source_change'})
add('adv','cuda_jl','work_item_mapping','One scalar coefficient per workitem in each of seven tensor stages; 256-thread groups; stage ranges 6144/12288/24576/24576/3072/3072/3072 (24/48/96/96/12/12/12 groups).','Stale pixel/16x4/320-block text contradicts bound tensor-stage source and existing correct launch_geometry.',['src/adv-julia/adv-julia-cuda.jl/main.jl'])
for m in ['cuda_jl','ka','jacc']:
 add('bilateral',m,'work_item_mapping','One output pixel per workitem, with 16x16 workitems per workgroup; each workgroup covers up to 256 pixels.','Current text says one pixel per whole workgroup, conflating lane and group.',[f'src/bilateral-julia/bilateral-julia-{dict(cuda_jl="cuda.jl",ka="KA",jacc="JACC")[m]}/main.jl'])
 add('bilateral',m,'launch_geometry',{'mapping':'One output pixel per workitem in a 16x16 workgroup','native_mapping_changed':False},'Same lane/group correction in nested geometry display.',[f'src/bilateral-julia/bilateral-julia-{dict(cuda_jl="cuda.jl",ka="KA",jacc="JACC")[m]}/main.jl'])
 for n in ['axhelm','amgmk','bh']:
  value=x[n]['implementations'][m]['normalized'].get('timing_scope')or x[n]['implementations'][m]['shared_traits'].get('timing_scope')
  add(n,m,'timing_scope',value,'Expose already verified selected-contract scope in graph/detail; do not substitute a kernel sample for elapsed pipeline.',['src/axhelm-julia/study_common.jl']if n=='axhelm'else[])
overlay={'schema_version':1,'status':'PRIVATE_PROPOSED_DISPLAY_OVERLAY_NOT_ADOPTED','scope':'Current537 selected bounded contracts only; no frozen normalized_records/cohort/base-YAML mutation','root_adoption':{'path':'ontology/studies/v100_roofline_537/completion_reviews/roi_membership_root_merge_v3/root_adoption.json','sha256':sha((R/'ontology/studies/v100_roofline_537/completion_reviews/roi_membership_root_merge_v3/root_adoption.json').read_bytes())},'display_overrides':patches,'source_evidence':sources,'selected_capture_bindings':captures,'sample_phase':'unknown_not_promoted','complete_elapsed_ROI_capture_claim':False,'numerical_outcomes_changed':False,'causal_claim':False}
(O/'proposed_current537_display_overlay.json').write_text(json.dumps(overlay,indent=2)+'\n');(O/'final_bindings_to_recheck.json').write_text(json.dumps(bindings,indent=2)+'\n')
print(json.dumps({'current_selected_models':len(captures),'display_overrides':len(patches),'source_excerpt_files':len(sources)}))
