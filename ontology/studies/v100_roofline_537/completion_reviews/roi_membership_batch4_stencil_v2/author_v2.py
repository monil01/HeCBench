from pathlib import Path
import json,hashlib,gzip,re,fcntl,copy,datetime
R=Path('/home/imo/HeCBench_julia');OLD=Path('/tmp/roofline-completion-20261021/roi_membership_batch4_stencil');O=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=OLD/'proposed_stage_patches.json';assert sha(p)=='6a1e6fad7f611048b27fcc3078ae8a5918a9b1b2588c3742dc7c94e0cecf3744';d=json.loads(p.read_text());reviews=[];kept=[];held=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for record in d['patches']:
  if record['record_key']['benchmark']!='clink':kept.append(record);continue
  m=record['record_key']['model']
  if m=='cuda':kept.append(record);continue
  capture=next(x for x in d['selected_capture_bindings']if x['benchmark']=='clink'and x['model']==m)
  pp=Path(capture['plan']['path']);assert sha(pp)==capture['plan']['sha256'];plan=json.loads(gzip.decompress(pp.read_bytes()));op=Path(capture['derivation']['path']);assert sha(op)==capture['derivation']['sha256'];obs=json.loads(gzip.decompress(op.read_bytes()))['records'];assert len(obs)==1;identity=obs[0]['identity'];name=identity['Kernel Name']
  entry=R/'src/clink-julia'/('clink-julia-cuda.jl'if m=='cuda_jl'else'clink-julia-KA'if m=='ka'else'clink-julia-JACC')/'main.jl';common=R/'src/clink-julia/study_common.jl';body=R/'src/clink-julia/study_device.jl';source=[]
  for path in [entry,common,body]:
   assert plan['execution_bindings'][str(path)]==sha(path)
   source.append({'path':str(path),'sha256':sha(path),'snippet':path.read_text(),'start_line':1,'end_line':path.read_text().count('\n')+1})
  proof={'scope':'Exact source kernel/callback argument identity, separate from sampled invocation phase','model':m,'sources':source,'selected_plan':capture['plan'],'selected_derivation':capture['derivation'],'saved_kernel_identity':identity,'source_kernel_parameter_contract':'kernel(x,w,y), all three CuDeviceArrayFloat32 buffers; lstm_lane! consumes x,w and writes y; factory returned closure supplied to shared timer.','phase':'unknown','SASS_or_target_binary_inspection':'not available in selected evidence; no new compilation/run requested'}
  if m=='ka':
   proof.update(decision='withhold_selected_counter_to_math_family_mapping',reason='Saved gpu_kernel signature contains CompilerMetadata,oneCuDeviceArrayFloat32,twoFloat32 scalars; source mathematical kernel receives threeCuDeviceArrayFloat32 buffers. Existing proof does not establish parameter-lowering equivalence or identify another source of this kernel. No fill/setup/math or compiler-bug conclusion made.')
   held.append({'record_key':record['record_key'],'selected_capture_binding':capture,'reason':proof['reason'],'source_identity_review':proof,'inclusion_untouched':True,'future_requirement':'Exact selected-instance compiled identity/source proof or separately reviewed fresh profiling; never broaden generic gpu_kernel regex merely to create positive match.'});reviews.append(proof);continue
  expected_cuda='kernel(CuDeviceArray<Float32, (int)1, (int)1>, CuDeviceArray<Float32, (int)1, (int)1>, CuDeviceArray<Float32, (int)1, (int)1>)'
  expected_jacc='_parallel_for_cuda(Int64, kernel, CuDeviceArray<Float32, (int)1, (int)1>, CuDeviceArray<Float32, (int)1, (int)1>, CuDeviceArray<Float32, (int)1, (int)1>)'
  assert name==(expected_cuda if m=='cuda_jl'else expected_jacc)
  text=entry.read_text()
  assert 'lstm_lane!(x,w,y,'in text
  if m=='cuda_jl':assert '@cuda threads=256 blocks=2 kernel(x,w,y)'in text
  else:assert 'JACC.parallel_for(SPEC,512,kernel,x,w,y)'in text
  assert 'push!(launches,factory(x,w,y))'in common.read_text()and 'for r in 1:10;f();end'in common.read_text()
  selector='^'+re.escape(name)+'$';sp=record['stage_patches'][0];sp['updates'].update(model_kernel_regex=selector,regex_status='exact_saved_argument_signature_bound_to_explicit_source_callback_not_generic_wrapper_match',resolved_callable_candidates=['kernel'])
  proof.update(decision='exact_source_callback_and_three_array_signature_match',selector=selector,reason='The source explicitly passes the kernel callback/three arrays; saved signature matches all three buffer types and argument order. clink_study is a host factory wrapper, not the launched mathematical kernel. This exact alias is not inferred from timing/counters or broadened from a generic dispatcher.')
  sp['updates']['ROI_membership_proof']['source_kernel_identity_review']=proof
  record['selected_observation_matching']=[{'identity':identity,'matched_stage_ids':[sp['stage_id']],'runtime_first_sample_phase':'unproved','source_family_membership_only':True}]
  kept.append(record);reviews.append(proof)
 # Parent/source/capture inputs remain byte-frozen; do not rewrite changed selections.
 for path,v in d['parent_reviews'].items():assert sha(path)==v['compressed_sha256']
 for x in d['sources']:assert sha(x['path'])==x['sha256']
 for x in d['selected_capture_bindings']:
  for key in ['ledger','manifest','plan','derivation']:assert sha(x[key]['path'])==x[key]['sha256']
d['patches']=kept;d['held']=held;d['scope']='Private batch4 v2:ten names39positive-model records; clinkKA exactcounter→source identity unresolved and withheld';d['supersedes_private_proposal']={'path':str(p),'sha256':sha(p),'historical_bytes_preserved':True,'reason':'Root positive-savedmatch dryruns rejected clinkJulia3; exactCUDAjl/JACCcallback-signature proof fixes2, KA withheld1. No source/counter/timing changes.'};d['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['withheld_models']=[{'benchmark':'clink','model':'ka','reason':held[0]['reason']}]
(O/'proposed_stage_patches.json').write_text(json.dumps(d,indent=2)+'\n');(O/'clink_identity_review.json').write_text(json.dumps({'reviews':reviews,'no_GPU_build_tests_imports':True},indent=2)+'\n')
for name in ['scientific_observations.json','scientific_observations.tex']:
 dest=O/name
 if name.endswith('.json'):
  x=json.loads((OLD/name).read_text());x['proposed_patch_sha256']=sha(O/'proposed_stage_patches.json');dest.write_text(json.dumps(x,indent=2)+'\n')
 else:dest.write_bytes((OLD/name).read_bytes())
print(json.dumps({'records':len(kept),'stages':sum(len(r['stage_patches'])for r in kept),'held':len(held),'patch_sha256':sha(O/'proposed_stage_patches.json')}))
